// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import { stemmer } from "stemmer";

import { stopWords as defaultStopWords } from "./stop_words";

export interface Rectangle {
  xMin: number;
  xMax: number;
  yMin: number;
  yMax: number;
}

/** A text summarizer based on c-TF-IDF (https://arxiv.org/pdf/2203.05794) */
export class TextSummarizer {
  private segmenter: Intl.Segmenter;
  private binning: XYBinning;
  private stopWords: Set<string>;
  private key2RegionIndices: Map<number, number[]>;
  private frequencyPerClass: Map<string, number>[];
  private frequencyAll: Map<string, number>;

  /** Create a new TextSummarizer */
  constructor(options: { regions: Rectangle[][]; stopWords?: string[] | null }) {
    this.binning = XYBinning.inferFromRegions(options.regions);
    this.segmenter = new Intl.Segmenter(undefined, { granularity: "word" });
    this.stopWords = new Set(options.stopWords ?? defaultStopWords);

    this.frequencyPerClass = options.regions.map(() => new Map());
    this.frequencyAll = new Map();

    // Generate key2RegionIndices, a map from xy key to region index
    this.key2RegionIndices = new Map();
    for (let i = 0; i < options.regions.length; i++) {
      const keys = this.binning.keys(options.regions[i]);
      for (const k of keys) {
        const v = this.key2RegionIndices.get(k);
        if (v != null) {
          v.push(i);
        } else {
          this.key2RegionIndices.set(k, [i]);
        }
      }
    }
  }

  /** Add data to the summarizer */
  add(data: { x: ArrayLike<number>; y: ArrayLike<number>; text: ArrayLike<string> }) {
    for (let i = 0; i < data.text.length; i++) {
      const key = this.binning.key(data.x[i], data.y[i]);
      const indices = this.key2RegionIndices.get(key);
      if (indices == null) {
        continue;
      }
      const words = [];
      for (const s of this.segmenter.segment(data.text[i])) {
        const word = s.segment.trim();
        if (word.length > 1) {
          words.push(word);
        }
      }
      const inc = 1 / words.length;
      for (const word of words) {
        for (const idx of indices) {
          incrementMap(this.frequencyPerClass[idx], word, inc);
        }
        incrementMap(this.frequencyAll, word, inc);
      }
    }
  }

  summarize(limit: number = 4): string[][] {
    // Aggregate the frequencies by stemmed words
    const frequencyAllStem = aggregateByStem(this.frequencyAll, this.stopWords);
    const frequencyPerClassStem = this.frequencyPerClass.map((m) => aggregateByStem(m, this.stopWords));

    // Average number of words per class
    const averageWords =
      frequencyPerClassStem.map((x) => x.values().reduce((a, b) => a + b[1], 0)).reduce((a, b) => a + b, 0) /
      frequencyPerClassStem.length;

    return frequencyPerClassStem.map((wordMap) => {
      // Compute TF-IDF
      let entries = Array.from(
        wordMap.entries().map(([key, [word, tf]]) => {
          const df = frequencyAllStem.get(key)?.[1] ?? 1;
          const idf = Math.log(1 + averageWords / df);
          return {
            word: word,
            tf: tf,
            df: df,
            idf: idf,
            tfIDF: tf * idf,
          };
        }),
      );
      entries = entries.filter((x) => x.df >= 2);
      entries = entries.sort((a, b) => b.tfIDF - a.tfIDF);
      return entries.slice(0, limit).map((x) => x.word);
    });
  }
}

class XYBinning {
  private xMin: number;
  private yMin: number;
  private xStep: number;
  private yStep: number;

  constructor(xMin: number, yMin: number, xStep: number, yStep: number) {
    this.xMin = xMin;
    this.yMin = yMin;
    this.xStep = xStep;
    this.yStep = yStep;
  }

  static inferFromRegions(regions: Rectangle[][]): XYBinning {
    let xMin = Number.POSITIVE_INFINITY;
    let yMin = Number.POSITIVE_INFINITY;
    let xMax = Number.NEGATIVE_INFINITY;
    let yMax = Number.NEGATIVE_INFINITY;
    for (const region of regions) {
      for (const rect of region) {
        if (rect.xMin < xMin) {
          xMin = rect.xMin;
        } else if (rect.xMax > xMax) {
          xMax = rect.xMax;
        }
        if (rect.yMin < yMin) {
          yMin = rect.yMin;
        } else if (rect.yMax > yMax) {
          yMax = rect.yMax;
        }
      }
    }
    if (xMin < xMax && yMin < yMax) {
      return new XYBinning(xMin, yMin, (xMax - xMin) / 200, (yMax - yMin) / 200);
    } else {
      return new XYBinning(0, 0, 1, 1);
    }
  }

  key(x: number, y: number) {
    const ix = Math.floor((x - this.xMin) / this.xStep);
    const iy = Math.floor((y - this.yMin) / this.yStep);
    return ix + iy * 32768;
  }

  keys(rects: Rectangle[]): Set<number> {
    const keys = new Set<number>();
    for (const { xMin, yMin, xMax, yMax } of rects) {
      const xiLowerBound = Math.floor((xMin - this.xMin) / this.xStep);
      const xiUpperBound = Math.floor((xMax - this.xMin) / this.xStep);
      const yiLowerBound = Math.floor((yMin - this.yMin) / this.yStep);
      const yiUpperBound = Math.floor((yMax - this.yMin) / this.yStep);
      for (let xi = xiLowerBound; xi <= xiUpperBound; xi++) {
        for (let yi = yiLowerBound; yi <= yiUpperBound; yi++) {
          const p = yi * 32768 + xi;
          keys.add(p);
        }
      }
    }
    return keys;
  }
}

function incrementMap<K>(map: Map<K, number>, key: K, value: number) {
  const c = map.get(key) ?? 0;
  map.set(key, c + value);
}

/** Aggregate words by their stems and track the most frequent version.
 * Returns a map with stemmed words as keys, and the most frequent version and total count as values. */
function aggregateByStem(inputMap: Map<string, number>, stopWords: Set<string>): Map<string, [string, number]> {
  const result = new Map();
  for (const [word, count] of inputMap.entries()) {
    const lower = word.toLowerCase();
    if (stopWords.has(lower) || /^[0-9]+$/.test(lower)) {
      continue;
    }
    const s = stemmer(lower);
    if (result.has(s)) {
      const value = result.get(s);
      value[1] += count;
      if ((inputMap.get(value[0]) ?? 0) < count) {
        value[0] = word;
      }
    } else {
      result.set(s, [word, count]);
    }
  }
  return result;
}
