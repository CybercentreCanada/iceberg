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

import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import org.apache.iceberg.Schema;
import org.apache.iceberg.expressions.Binder;
import org.apache.iceberg.expressions.Bound;
import org.apache.iceberg.expressions.BoundReference;
import org.apache.iceberg.expressions.Expression;
import org.apache.iceberg.expressions.ExpressionVisitors;
import org.apache.iceberg.expressions.ExpressionVisitors.BoundExpressionVisitor;
import org.apache.iceberg.expressions.Expressions;
import org.apache.iceberg.expressions.Literal;
import org.apache.iceberg.types.Types.StructType;
import org.apache.parquet.column.page.DictionaryPageReadStore;
import org.apache.parquet.hadoop.BloomFilterReader;
import org.apache.parquet.hadoop.metadata.BlockMetaData;
import org.apache.parquet.schema.MessageType;

public class ParquetCombinedRowGroupFilter {
  private final Schema schema;
  private final Expression expr;
  private final ParquetMetricsRowGroupFilter statsFilter;
  private final ParquetDictionaryRowGroupFilter dictFilter;
  private final ParquetBloomRowGroupFilter bloomFilter;

  public ParquetCombinedRowGroupFilter(Schema schema, Expression unbound) {
    this(schema, unbound, true);
  }

  public ParquetCombinedRowGroupFilter(Schema schema, Expression unbound, boolean caseSensitive) {
    this.schema = schema;
    this.statsFilter = new ParquetMetricsRowGroupFilter(schema, unbound, caseSensitive);
    this.dictFilter = new ParquetDictionaryRowGroupFilter(schema, unbound, caseSensitive);
    this.bloomFilter = new ParquetBloomRowGroupFilter(schema, unbound, caseSensitive);
    StructType struct = schema.asStruct();
    this.expr = Binder.bind(struct, Expressions.rewriteNot(unbound), caseSensitive);
  }

  /**
   * Test whether the dictionaries for a row group may contain records that match the expression.
   *
   * @param fileSchema schema for the Parquet file
   * @param dictionaries a dictionary page read store
   * @return false if the file cannot contain rows that match the expression, true otherwise.
   */
  public boolean shouldRead(
      MessageType fileSchema,
      BlockMetaData rowGroup,
      DictionaryPageReadStore dictionaries,
      BloomFilterReader bloomReader) {
    List<ParquetRowGroupEvaluator> evaluators = new ArrayList();

    ParquetRowGroupEvaluator eval = null;

    eval = statsFilter.buildVisitor(fileSchema, rowGroup);
    if (eval.getInitStatus() == ROWS_CANNOT_MATCH) {
      return ROWS_CANNOT_MATCH;
    }
    evaluators.add(eval);

    eval = dictFilter.buildVisitor(fileSchema, rowGroup, dictionaries);
    if (eval.getInitStatus() == ROWS_CANNOT_MATCH) {
      return ROWS_CANNOT_MATCH;
    }
    evaluators.add(eval);

    eval = bloomFilter.buildVisitor(fileSchema, rowGroup, bloomReader);
    if (eval.getInitStatus() == ROWS_CANNOT_MATCH) {
      return ROWS_CANNOT_MATCH;
    }
    evaluators.add(eval);

    return new CombinedEvalVisitor(evaluators).eval();
  }

  private static final boolean ROWS_MIGHT_MATCH = true;
  private static final boolean ROWS_CANNOT_MATCH = false;

  private class CombinedEvalVisitor extends BoundExpressionVisitor<Boolean> {
    private final List<ParquetRowGroupEvaluator> evaluators;

    private CombinedEvalVisitor(List<ParquetRowGroupEvaluator> evaluators) {
      this.evaluators = evaluators;
    }

    private boolean eval() {
      return ExpressionVisitors.visitEvaluator(expr, this);
    }

    @Override
    public Boolean alwaysTrue() {
      return ROWS_MIGHT_MATCH; // all rows match
    }

    @Override
    public Boolean alwaysFalse() {
      return ROWS_CANNOT_MATCH; // all rows fail
    }

    @Override
    public Boolean not(Boolean result) {
      return !result;
    }

    @Override
    public Boolean and(Boolean leftResult, Boolean rightResult) {
      return leftResult && rightResult;
    }

    @Override
    public Boolean or(Boolean leftResult, Boolean rightResult) {
      return leftResult || rightResult;
    }

    // Evaluations of BoundReferences is delegated to the metrics, dictionary and bloom evaluators.

    @Override
    public <T> Boolean isNull(BoundReference<T> ref) {
      return evaluators.stream().allMatch(v -> v.isNull(ref) == ROWS_MIGHT_MATCH);
    }

    @Override
    public <T> Boolean notNull(BoundReference<T> ref) {
      return evaluators.stream().allMatch(v -> v.notNull(ref) == ROWS_MIGHT_MATCH);
    }

    @Override
    public <T> Boolean isNaN(BoundReference<T> ref) {
      return evaluators.stream().allMatch(v -> v.isNaN(ref) == ROWS_MIGHT_MATCH);
    }

    @Override
    public <T> Boolean notNaN(BoundReference<T> ref) {
      return evaluators.stream().allMatch(v -> v.notNaN(ref) == ROWS_MIGHT_MATCH);
    }

    @Override
    public <T> Boolean lt(BoundReference<T> ref, Literal<T> lit) {
      return evaluators.stream().allMatch(v -> v.lt(ref, lit) == ROWS_MIGHT_MATCH);
    }

    @Override
    public <T> Boolean ltEq(BoundReference<T> ref, Literal<T> lit) {
      return evaluators.stream().allMatch(v -> v.ltEq(ref, lit) == ROWS_MIGHT_MATCH);
    }

    @Override
    public <T> Boolean gt(BoundReference<T> ref, Literal<T> lit) {
      return evaluators.stream().allMatch(v -> v.gt(ref, lit) == ROWS_MIGHT_MATCH);
    }

    @Override
    public <T> Boolean gtEq(BoundReference<T> ref, Literal<T> lit) {
      return evaluators.stream().allMatch(v -> v.gtEq(ref, lit) == ROWS_MIGHT_MATCH);
    }

    @Override
    public <T> Boolean eq(BoundReference<T> ref, Literal<T> lit) {
      return evaluators.stream().allMatch(v -> v.eq(ref, lit) == ROWS_MIGHT_MATCH);
    }

    @Override
    public <T> Boolean in(BoundReference<T> ref, Set<T> literalSet) {
      return evaluators.stream().allMatch(v -> v.in(ref, literalSet) == ROWS_MIGHT_MATCH);
    }

    @Override
    public <T> Boolean notIn(BoundReference<T> ref, Set<T> literalSet) {
      return evaluators.stream().allMatch(v -> v.notIn(ref, literalSet) == ROWS_MIGHT_MATCH);
    }

    @Override
    public <T> Boolean startsWith(BoundReference<T> ref, Literal<T> lit) {
      return evaluators.stream().allMatch(v -> v.startsWith(ref, lit) == ROWS_MIGHT_MATCH);
    }

    @Override
    public <T> Boolean notStartsWith(BoundReference<T> ref, Literal<T> lit) {
      return evaluators.stream().allMatch(v -> v.notStartsWith(ref, lit) == ROWS_MIGHT_MATCH);
    }

    @Override
    public <T> Boolean handleNonReference(Bound<T> term) {
      return evaluators.stream().allMatch(v -> v.handleNonReference(term) == ROWS_MIGHT_MATCH);
    }
  }
}
