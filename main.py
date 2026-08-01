"""Demo rapido de doekit. Ejecutar: uv run python main.py"""

import doekit as ed


def main():
    print("doekit", ed.__version__)

    print("\n== Screening (Plackett-Burman, 6 factores) ==")
    pb = ed.plackett_burman(6)
    print(f"{pb.n_runs} corridas, orden Hadamard {pb.metadata['order']}, "
          f"PB valido: {ed.is_plackett_burman(pb)}")

    print("\n== Fraccional real 2^(3-1) ==")
    fr = ed.fractional_factorial(3, generators=["C=AB"])
    print(f"{fr.n_runs} corridas, resolucion {fr.metadata['resolution']}, "
          f"{fr.metadata['defining_relation']}")

    print("\n== Superficie de respuesta (CCD, 3 factores) ==")
    cc = ed.central_composite(3)
    print(f"{cc.n_runs} corridas, alpha = {cc.metadata['alpha_value']:.4f}")

    print("\n== Diseno D-optimo (KL-exchange) ==")
    cand = ed.full_factorial({f"factor{i+1}": [-1, 0, 1] for i in range(5)})
    cand.model = ed.Model.parse(
        "0 ~ factor1 + factor2 + factor3 + factor4 + factor5 + factor3^2"
    )
    opt = ed.optimal_design(cand, n_runs=11, criterion="D", n_starts=5, seed=1)
    print(f"D = {opt.metadata['criteria']['D']:.4f} (11 de {cand.n_runs} candidatos)")


if __name__ == "__main__":
    main()
