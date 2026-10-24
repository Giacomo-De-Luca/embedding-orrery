# 2D density label fields

The 2D density overlay labels each detected blob using the majority value of
the selected categorical color field. For example, selecting `source` shows
source names even when those documents also have `topic_label` metadata.
Previously, the pipeline always preferred topics whenever the collection had
them, so its titles and the selected coloring could disagree.

`ScatterPlot2D` supplies `labelField` to `useDensityClusterLabels`. For numeric
color scales or no selected field, the existing topic-label fallback remains;
collections without topics use document keywords. A blob with no nonmissing
values for the chosen field also uses document keywords, without silently
switching to a different metadata field. Ties are resolved lexicographically.

`resolveClusterLabelTexts` in `app/utils/densityClusterLabels.ts` returns
`categoryLabel` separately from `topicLabel`. The former chooses the label's
color; the latter enables topic selection only for labels derived from
`topic_label`. Source names and keyword summaries cannot trigger topic clicks
even if their text matches a topic. Nested topic/subtopic colors are consulted
only for the corresponding field.

When the points or the field change, the hook stops showing titles made from
the old ones while its debounced clustering runs. The plot can show current
category-centroid labels during that interval. Placement and colour are a
separate step on the latest titles (`DensityLabelColors.resolve` picks the
colour), so a new plot area, label font, palette or theme re-places and
recolours the labels without clustering again (`PLOT_LABELS.md`). Blob
detection, density thresholds, and title deduplication are unchanged.

## Verification

Run from `embedding_visualization/`:

```sh
npm run test:run -- app/utils/__tests__/densityClusterLabels.test.ts lib/utils/densityRenderer/__tests__/densityMath.test.ts lib/utils/__tests__/categoryKey.test.ts
```

The label tests cover source/topic switching, subtopics, coincident source/topic
names, missing values, numeric zero as a category, keyword fallback, category
colors, and the existing density/placement behavior.
