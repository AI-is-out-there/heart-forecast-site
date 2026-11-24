/* Точный TreeSHAP для экспортированного XGBoost Cox. В каждом дереве не больше семи
 * признаков, поэтому значения Шепли вычисляются прямым перебором коалиций. Веса узлов —
 * доли обучающих пациентов, проходящих через узел (tree_path_dependent).
 */
(function (root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  else root.TreeSHAP = api;
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  function create(bundle, background) {
    const { features, scaler, models, patients } = bundle;
    const rows = background || patients.X.filter((_, i) => patients.split[i] === 1);
    const scale = (x) => Float64Array.from(x, (v, k) => (v - scaler.mean[k]) / scaler.scale[k]);
    const Z = rows.map(scale);
    const trees = models.xgb.trees.map((tree) => ({
      left: Int32Array.from(tree.left),
      right: Int32Array.from(tree.right),
      feature: Int32Array.from(tree.feature),
      value: Float64Array.from(tree.value, Math.fround),
      cover: new Float64Array(tree.left.length),
    }));

    for (const tree of trees) {
      for (const z of Z) {
        let node = 0;
        for (;;) {
          tree.cover[node]++;
          if (tree.left[node] === -1) break;
          node = Math.fround(z[tree.feature[node]]) < tree.value[node] ? tree.left[node] : tree.right[node];
        }
      }
    }

    const factorial = [1, 1, 2, 6, 24, 120, 720, 5040];
    const popcount = (mask) => {
      let n = 0;
      for (; mask; mask &= mask - 1) n++;
      return n;
    };

    function treeValues(tree, z, out) {
      const localFeatures = [];
      const localIndex = new Int32Array(features.length).fill(-1);
      for (let i = 0; i < tree.left.length; i++) {
        const f = tree.feature[i];
        if (tree.left[i] !== -1 && localIndex[f] < 0) {
          localIndex[f] = localFeatures.length;
          localFeatures.push(f);
        }
      }
      const m = localFeatures.length, size = 1 << m;
      const coalitionValue = new Float64Array(size);

      for (let mask = 0; mask < size; mask++) {
        function expected(node) {
          if (tree.left[node] === -1) return tree.value[node];
          const f = tree.feature[node];
          const bit = 1 << localIndex[f];
          if (mask & bit) {
            const next = Math.fround(z[f]) < tree.value[node] ? tree.left[node] : tree.right[node];
            return expected(next);
          }
          return (tree.cover[tree.left[node]] * expected(tree.left[node])
            + tree.cover[tree.right[node]] * expected(tree.right[node])) / tree.cover[node];
        }
        coalitionValue[mask] = expected(0);
      }

      for (let j = 0; j < m; j++) {
        const bit = 1 << j;
        let contribution = 0;
        for (let mask = 0; mask < size; mask++) {
          if (mask & bit) continue;
          const n = popcount(mask);
          const weight = factorial[n] * factorial[m - n - 1] / factorial[m];
          contribution += weight * (coalitionValue[mask | bit] - coalitionValue[mask]);
        }
        out[localFeatures[j]] += contribution;
      }
      return { expected: coalitionValue[0], actual: coalitionValue[size - 1] };
    }

    let baseValue = models.xgb.bias;
    const zero = new Float64Array(features.length);
    for (const tree of trees) baseValue += treeValues(tree, zero, new Float64Array(features.length)).expected;
    if (Math.abs(baseValue) < 1e-12) baseValue = 0;

    function explain(x) {
      const z = scale(x), values = new Float64Array(features.length);
      for (const tree of trees) treeValues(tree, z, values);
      return { baseValue, values, score: baseValue + values.reduce((sum, v) => sum + v, 0) };
    }

    return { explain, baseValue, backgroundSize: rows.length };
  }

  return { create };
});
