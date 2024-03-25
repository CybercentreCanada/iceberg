/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
package org.apache.iceberg.parquet;

import static org.apache.iceberg.TableProperties.PARQUET_BLOOM_FILTER_COLUMN_ENABLED_PREFIX;
import static org.apache.iceberg.TableProperties.PARQUET_BLOOM_FILTER_MAX_BYTES;
import static org.apache.iceberg.expressions.Expressions.equal;
import static org.apache.iceberg.expressions.Expressions.or;
import static org.apache.iceberg.parquet.ParquetWritingTestUtils.createTempFile;
import static org.assertj.core.api.Assertions.assertThat;

import java.io.File;
import java.nio.file.Path;
import java.util.List;
import org.apache.iceberg.Files;
import org.apache.iceberg.PartitionSpec;
import org.apache.iceberg.Schema;
import org.apache.iceberg.SortOrder;
import org.apache.iceberg.data.GenericRecord;
import org.apache.iceberg.data.Record;
import org.apache.iceberg.data.parquet.GenericParquetWriter;
import org.apache.iceberg.io.DataWriter;
import org.apache.iceberg.io.InputFile;
import org.apache.iceberg.io.OutputFile;
import org.apache.iceberg.relocated.com.google.common.collect.ImmutableList;
import org.apache.iceberg.relocated.com.google.common.collect.ImmutableMap;
import org.apache.iceberg.types.Types;
import org.apache.parquet.column.page.DictionaryPageReadStore;
import org.apache.parquet.hadoop.BloomFilterReader;
import org.apache.parquet.hadoop.ParquetFileReader;
import org.apache.parquet.hadoop.ParquetOutputFormat;
import org.apache.parquet.hadoop.metadata.BlockMetaData;
import org.apache.parquet.schema.MessageType;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

public class TestCombinedRowGroupFilter {

  private static final Schema SCHEMA =
      new Schema(
          Types.NestedField.required(1, "id", Types.LongType.get()),
          Types.NestedField.required(2, "category", Types.LongType.get()),
          Types.NestedField.optional(3, "data", Types.StringType.get()));

  private List<Record> records;

  @TempDir private Path temp;

  private MessageType parquetSchema = null;
  private BlockMetaData rowGroupMetadata = null;
  private DictionaryPageReadStore dictionaryStore = null;
  private BloomFilterReader bloomStore = null;

  @BeforeEach
  public void createRecords() throws Exception {
    GenericRecord record = GenericRecord.create(SCHEMA);

    ImmutableList.Builder<Record> builder = ImmutableList.builder();
    for (long i = 0; i < 1000; i++) {
      long id = i * 2; // id has many even numbers 0, 2, 4, 6, 8... 2000
      long category = (i % 3) * 2; // category has few even numbers 0, 2, 4
      builder.add(record.copy(ImmutableMap.of("id", id, "category", category, "data", "somedata")));
    }
    this.records = builder.build();

    File tmpFile = createTempFile(temp);
    OutputFile file = Files.localOutput(tmpFile);

    SortOrder sortOrder = SortOrder.builderFor(SCHEMA).withOrderId(10).asc("id").build();

    DataWriter<Record> dataWriter =
        Parquet.writeData(file)
            .schema(SCHEMA)
            .set(PARQUET_BLOOM_FILTER_MAX_BYTES, "256")
            .set(ParquetOutputFormat.ENABLE_DICTIONARY, "false")
            .set(PARQUET_BLOOM_FILTER_COLUMN_ENABLED_PREFIX + "id", "true")
            .set(PARQUET_BLOOM_FILTER_COLUMN_ENABLED_PREFIX + "category", "true")
            .set(PARQUET_BLOOM_FILTER_COLUMN_ENABLED_PREFIX + "data", "true")
            .createWriterFunc(GenericParquetWriter::buildWriter)
            .overwrite()
            .withSpec(PartitionSpec.unpartitioned())
            .withSortOrder(sortOrder)
            .build();

    try {
      for (Record rec : records) {
        dataWriter.write(rec);
      }
    } finally {
      dataWriter.close();
    }

    InputFile inFile = Files.localInput(tmpFile);

    ParquetFileReader reader = ParquetFileReader.open(ParquetIO.file(inFile));

    assertThat(reader.getRowGroups()).as("Should create only one row group").hasSize(1);
    rowGroupMetadata = reader.getRowGroups().get(0);
    parquetSchema = reader.getFileMetaData().getSchema();
    dictionaryStore = reader.getNextDictionaryReader();
    bloomStore = reader.getBloomFilterDataReader(rowGroupMetadata);
  }

  @Test
  public void testAssumptions() {
    boolean shouldRead = true;
    shouldRead =
        new ParquetMetricsRowGroupFilter(SCHEMA, equal("id", 10000))
            .shouldRead(parquetSchema, rowGroupMetadata);
    assertThat(shouldRead).as("id=10000 does not exists and is out of range of metrics").isFalse();
    shouldRead =
        new ParquetMetricsRowGroupFilter(SCHEMA, equal("id", 1))
            .shouldRead(parquetSchema, rowGroupMetadata);
    assertThat(shouldRead).as("id=1 does not exists but is in range of metrics").isTrue();

    shouldRead =
        new ParquetBloomRowGroupFilter(SCHEMA, equal("id", 10000))
            .shouldRead(parquetSchema, rowGroupMetadata, bloomStore);
    assertThat(shouldRead).as("id=10000 does not exists, however bloom is saturated").isTrue();

    shouldRead =
        new ParquetMetricsRowGroupFilter(SCHEMA, equal("category", 100))
            .shouldRead(parquetSchema, rowGroupMetadata);
    assertThat(shouldRead)
        .as("category=100 does not exists and is out of range of metrics")
        .isFalse();
    shouldRead =
        new ParquetMetricsRowGroupFilter(SCHEMA, equal("category", 1))
            .shouldRead(parquetSchema, rowGroupMetadata);
    assertThat(shouldRead).as("category=1 does not exists but is in range of metrics").isTrue();
    shouldRead =
        new ParquetBloomRowGroupFilter(SCHEMA, equal("category", 1))
            .shouldRead(parquetSchema, rowGroupMetadata, bloomStore);
    assertThat(shouldRead).as("category=1 does not exists and bloom is not saturated").isFalse();
  }

  @Test
  public void testIndependentaly() {
    // combining metrics && bloom indepently would return shouldRead=true
    boolean metricsShouldRead =
        new ParquetMetricsRowGroupFilter(SCHEMA, or(equal("id", 10000), equal("category", 1)))
            .shouldRead(parquetSchema, rowGroupMetadata);
    assertThat(metricsShouldRead)
        .as("id=10000 out of range, category=1 in range OR -> shouldRead=true")
        .isTrue();

    boolean bloomShouldRead =
        new ParquetBloomRowGroupFilter(SCHEMA, or(equal("id", 10000), equal("category", 1)))
            .shouldRead(parquetSchema, rowGroupMetadata, bloomStore);
    assertThat(bloomShouldRead)
        .as("id=10000 saturated bloom, category=1 not in bloom OR -> shouldRead=true")
        .isTrue();
  }

  @Test
  public void testCombined() {
    boolean combinedShouldRead =
        new ParquetCombinedRowGroupFilter(SCHEMA, or(equal("id", 1), equal("category", 100)))
            .shouldRead(parquetSchema, rowGroupMetadata, dictionaryStore, bloomStore);
    assertThat(combinedShouldRead)
        .as("id=10000 out of range, category=1 not in bloom OR -> shouldRead=false")
        .isFalse();
  }
}
