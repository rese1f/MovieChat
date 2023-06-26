MathJax.Hub.Config({
  jax: ["input/TeX", "output/HTML-CSS"],
  tex2jax: {
    inlineMath: [['$', '$']],
    displayMath: [['$$', '$$'],['\\[','\\]']],
    processEscapes: true,
    skipTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code']
  },
  messageStyle: "none",
  "HTML-CSS": { preferredFont: "TeX", availableFonts: ["STIX","TeX"] },
  TeX: {
    equationNumbers: { autoNumber: "AMS" },
    extensions: ["AMSmath.js", "AMSsymbols.js","AMScd.js"],
    TagSide: "left",
    Macros: {
      field: ['\\mathbb{#1}', 1],
      C: ['\\field{C}'],
      F: ['\\field{F}'],
      N: ['\\field{N}'],
      Q: ['\\field{Q}'],
      R: ['\\field{R}'],
      Z: ['\\field{Z}'],

      zeros: ['\\mathbf{0}'],
      ud: ['\\,\\mathrm{d}'],

      vect:['\\boldsymbol{\\mathbf{#1}}',1],
      abs: ['\\lvert#1\\rvert', 1],
      abslr:['\\left\\lvert#1\\right\\rvert', 1],
      norm: ['\\lVert#1\\rVert', 1],
      normlr: ['\\left\\lVert#1\\right\\rVert', 1],

      lcm: ['\\mathop{\\mathrm{lcm}}'],
      interior: ['\\mathop{\\mathrm{int}}'],
      exterior: ['\\mathop{\\mathrm{ext}}'],
      volume: ['\\mathop{\\mathrm{vol}}'],

      E: ['{\\rm I\\kern-.3em E}'],
      Var: ['\\mathop{\\mathrm{Var}}'],
      Cov: ['\\mathop{\\mathrm{Cov}}'],
      Binom: ['\\mathop{\\mathrm{Binom}}'],
      Exp: ['\\mathop{\\mathrm{Exp}}'],
      Poi: ['\\mathop{\\mathrm{Poi}}'],

      GL: ['\\mathrm{GL}'],
      SL: ['\\mathrm{SL}'],
      Aut: ['\\mathrm{Aut}'],
      ker: ['\\mathrm{ker}'],
      id: ['\\mathop{\\mathrm{id}}'],

      Re: ['\\mathop{\\mathrm{Re}}'],
      Im: ['\\mathop{\\mathrm{Im}}'],
      Res: ['\\mathop{\\mathrm{Res}}'],
    }
  }
});

MathJax.Ajax.loadComplete("https://idcrook.github.io/assets/js/MathJaxLocal.js");