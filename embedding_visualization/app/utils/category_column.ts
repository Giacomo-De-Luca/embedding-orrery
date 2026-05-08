// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import { defaultCategoryColors } from "./colors";
import type { Coordinator } from "@uwdata/mosaic-core";
import * as SQL from "@uwdata/mosaic-sql";
import { format } from "d3-format";

// import { inferBinning } from "../common/binning.js";
// import { defaultOrdinalColors } from "../common/colors.js";

// Mocking or simplifying dependencies for standalone utility
function jsTypeFromDBType(type: string): string {
    // Simplified mapping
    if (type.includes("INT") || type.includes("FLOAT") || type.includes("DOUBLE")) return "number";
    return "string";
}

function distinctCount(coordinator: Coordinator, table: string, column: string): Promise<number> {
    // Placeholder for actual implementation
    return Promise.resolve(100); 
}

export interface EmbeddingLegend {
  indexColumn: string;
  legend: {
    label: string;
    color: string;
    predicate: any;
    count: number;
  }[];
}

export async function makeCategoryColumn(
  coordinator: Coordinator,
  table: string,
  column: string | null | undefined,
): Promise<EmbeddingLegend | null> {
  if (column == null) {
    return null;
  }
  const [desc] = Array.from(await coordinator.query(SQL.Query.describe(SQL.Query.from(table).select(column))));
  if (desc == null) {
    return null;
  }
  const jsType = jsTypeFromDBType(desc.column_type);
  if (jsType == "string") {
    return await makeDiscreteCategoryColumn(coordinator, table, column, 10);
  } else if (jsType == "number") {
    // Simplified: always treat as discrete for now or implement binning if needed
    return await makeDiscreteCategoryColumn(coordinator, table, column, 10);
  }
  return null;
}

async function makeDiscreteCategoryColumn(
  coordinator: Coordinator,
  table: string,
  column: string,
  maxCategories: number,
): Promise<EmbeddingLegend> {
  const indexColumnName = `_ev_${column}_id`;
  const values = Array.from(
    await coordinator.query(
      SQL.Query.from(table)
        .select({ value: SQL.cast(SQL.column(column), "TEXT"), count: SQL.count() })
        .where(SQL.not(SQL.isNull(SQL.cast(SQL.column(column), "TEXT"))))
        .groupby(SQL.cast(SQL.column(column), "TEXT"))
        .orderby(SQL.desc(SQL.count()))
        .limit(maxCategories),
    ),
  ) as { value: string; count: number }[];

  const otherIndex = values.length;
  let nullIndex = values.length + 1;

  // Add the index column.
  await coordinator.exec(`
    ALTER TABLE ${table} ADD COLUMN IF NOT EXISTS ${SQL.column(indexColumnName)} INTEGER DEFAULT 0;
    UPDATE ${table}
    SET ${SQL.column(indexColumnName)} = CASE ${SQL.column(column)}::TEXT
          ${values.map(({ value }, i) => SQL.sql`WHEN ${SQL.literal(value)} THEN ${SQL.literal(i)}`).join(" ")}
          ELSE (CASE WHEN ${SQL.column(column)} IS NULL THEN ${SQL.literal(nullIndex)} ELSE ${SQL.literal(otherIndex)} END) END
  `);

  // Count by index.
  const counts = Array.from(
    await coordinator.query(
      SQL.Query.from(table)
        .select({ index: SQL.column(indexColumnName), count: SQL.cast(SQL.count(), "INT") })
        .groupby(SQL.column(indexColumnName)),
    ),
  );
  const countMap = new Map<number, number>();
  for (const item of counts) {
    countMap.set(item.index, item.count);
  }
  const otherCount = countMap.get(otherIndex) ?? 0;
  const nullCount = countMap.get(nullIndex) ?? 0;

  const colors = defaultCategoryColors(values.length);

  const legend: EmbeddingLegend["legend"] = values.map(({ value }, i) => ({
    label: value,
    color: colors[i],
    predicate: SQL.eq(SQL.cast(SQL.column(column), "TEXT"), SQL.literal(value)),
    count: countMap.get(i) ?? 0,
  }));

  if (otherCount > 0) {
    const { otherCategoryCount } = (
      await coordinator.query(`
        SELECT COUNT(DISTINCT(${SQL.column(column)}::TEXT)) AS otherCategoryCount
        FROM ${table}
        WHERE ${SQL.column(indexColumnName)} = ${SQL.literal(otherIndex)} AND ${SQL.column(column)} IS NOT NULL
    `)
    ).get(0);
    legend.push({
      label: `(other ${otherCategoryCount.toLocaleString()})`,
      color: "#9eabc2",
      predicate:
        values.length > 0
          ? SQL.sql`${SQL.column(column)} IS NOT NULL AND ${SQL.column(column)}::TEXT NOT IN (${values.map((x) => SQL.literal(x.value)).join(",")})`
          : SQL.sql`${SQL.column(column)} IS NOT NULL`,
      count: otherCount,
    });
  }
  if (nullCount > 0) {
    if (otherCount <= 0) {
      // If there is no other, reduce null index by 1 before we add the null item.
      await coordinator.exec(`
          UPDATE ${table}
          SET ${SQL.column(indexColumnName)} = ${SQL.column(indexColumnName)} - 1 WHERE ${SQL.column(indexColumnName)} = ${SQL.literal(nullIndex)}
        `);
      nullIndex -= 1;
    }
    legend.push({
      label: "(null)",
      color: "#aaaaaa",
      predicate: SQL.isNull(SQL.column(column)),
      count: nullCount,
    });
  }

  return {
    indexColumn: indexColumnName,
    legend: legend,
  };
}
