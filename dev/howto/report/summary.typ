#{
  table(
    align: (left, right, right, right, right, right, right, right,),
    columns: (auto, auto, auto, auto, auto, auto, auto, auto,),
    stroke: none,
    // == Horizontal Lines =================================================================
    table.hline(y: 0, stroke: 1.5pt,),
    table.hline(y: 1, stroke: 0.8pt,),
    table.hline(y: 2, stroke: 0.5pt,),
    table.hline(y: 5, stroke: 0.5pt,),
    table.hline(y: 6, stroke: 0.5pt,),
    table.hline(y: 7, stroke: 1.5pt,),
    // == Vertical Lines ===================================================================
    // == Table Header =====================================================================
    table.header(
      // -- Column Labels: Row 1 -----------------------------------------------------------
      [],
      [#text(weight: "bold",)[Estimate]],
      [#text(weight: "bold",)[Est.Error]],
      [#text(weight: "bold",)[l-95% CI]],
      [#text(weight: "bold",)[u-95% CI]],
      [#text(weight: "bold",)[Rhat]],
      [#text(weight: "bold",)[Bulk\_ESS]],
      [#text(weight: "bold",)[Tail\_ESS]],
    ),
    // == Table Body =======================================================================
    // -- Row Group Label: Row 1 -----------------------------------------------------------
    table.cell(align: left, colspan: 8,)[#text(weight: "bold",)[Regression Coefficients]],
    // -- Data: Row 1 ----------------------------------------------------------------------
    [#text(weight: "bold",)[Intercept]],
    [4.15],
    [0.27],
    [3.62],
    [4.68],
    [1.00],
    [4643],
    [3166],
    // -- Data: Row 2 ----------------------------------------------------------------------
    [#text(weight: "bold",)[light]],
    [0.56],
    [0.05],
    [0.47],
    [0.65],
    [1.00],
    [4478],
    [3031],
    // -- Data: Row 3 ----------------------------------------------------------------------
    [#text(weight: "bold",)[treatmentfertilised]],
    [1.66],
    [0.24],
    [1.18],
    [2.13],
    [1.00],
    [4217],
    [3016],
    // -- Row Group Label: Row 4 -----------------------------------------------------------
    table.cell(align: left, colspan: 8,)[
      #text(weight: "bold",)[Further Distributional Parameters]
    ],
    // -- Data: Row 4 ----------------------------------------------------------------------
    [#text(weight: "bold",)[sigma]],
    [1.19],
    [0.09],
    [1.03],
    [1.37],
    [1.00],
    [4501],
    [2768],
  )
}
