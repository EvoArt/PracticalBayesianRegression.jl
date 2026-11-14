#{
  table(
    align: (left, left, left, left, left, left,),
    columns: (auto, auto, auto, auto, auto, auto,),
    stroke: none,
    // == Horizontal Lines =================================================================
    table.hline(y: 0, stroke: 1.5pt,),
    table.hline(y: 1, stroke: 0.8pt,),
    table.hline(y: 5, stroke: 1.5pt,),
    // == Vertical Lines ===================================================================
    // == Table Header =====================================================================
    table.header(
      // -- Column Labels: Row 1 -----------------------------------------------------------
      [#text(weight: "bold",)[Prior]],
      [#text(weight: "bold",)[Class]],
      [#text(weight: "bold",)[Coef]],
      [#text(weight: "bold",)[Group]],
      [#text(weight: "bold",)[Dpar]],
      [#text(weight: "bold",)[Source]],
    ),
    // == Table Body =======================================================================
    // -- Data: Row 1 ----------------------------------------------------------------------
    [student\_t(3, 7.6, 2.5)],
    [Intercept],
    [],
    [],
    [],
    [default],
    // -- Data: Row 2 ----------------------------------------------------------------------
    [(flat)],
    [b],
    [light],
    [],
    [],
    [default],
    // -- Data: Row 3 ----------------------------------------------------------------------
    [(flat)],
    [b],
    [treatmentfertilised],
    [],
    [],
    [default],
    // -- Data: Row 4 ----------------------------------------------------------------------
    [student\_t(3, 0, 2.5)],
    [sigma],
    [],
    [],
    [],
    [default],
  )
}
