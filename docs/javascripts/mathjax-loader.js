(function () {
  const MATHJAX_URL =
    "https://unpkg.com/mathjax@3.2.2/es5/tex-mml-chtml.js";
  let mathJaxReady = null;

  window.MathJax = {
    tex: {
      inlineMath: [["\\(", "\\)"]],
      displayMath: [["\\[", "\\]"]],
      processEscapes: true,
      processEnvironments: true,
    },
    options: {
      ignoreHtmlClass: ".*|",
      processHtmlClass: "arithmatex",
    },
  };

  function typeset() {
    if (!window.MathJax?.typesetPromise) return;
    MathJax.startup.output.clearCache();
    MathJax.typesetClear();
    MathJax.texReset();
    return MathJax.typesetPromise();
  }

  function ensureMathJaxLoaded() {
    if (window.MathJax?.typesetPromise) {
      return Promise.resolve();
    }
    if (mathJaxReady) {
      return mathJaxReady;
    }

    mathJaxReady = new Promise((resolve, reject) => {
      const fail = () => {
        mathJaxReady = null;
        reject(new Error("MathJax failed to load"));
      };
      const finish = () => MathJax.startup.promise.then(resolve).catch(reject);

      let script = document.querySelector("script[data-mathjax]");
      if (!script) {
        script = document.createElement("script");
        script.src = MATHJAX_URL;
        script.async = true;
        script.dataset.mathjax = "true";
        script.onerror = fail;
        document.head.appendChild(script);
      }

      script.addEventListener("load", finish, { once: true });
      script.addEventListener("error", fail, { once: true });
    });

    return mathJaxReady;
  }

  function loadMathJax() {
    return ensureMathJaxLoaded().then(() => typeset());
  }

  document$.subscribe(() => {
    if (document.querySelector(".arithmatex")) {
      loadMathJax().catch((err) =>
        console.error("MathJax failed to load:", err)
      );
    }
  });
})();
