{
  description = "Apuntes MMI - LaTeX/beamer toolchain (pdflatex + latexmk)";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";

  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs { inherit system; };
      inherit (pkgs) lib;

      # texlive set covering every package used across the repo's .tex files
      # (collected from \usepackage across apuntes, tareas and Proyecto).
      tex = pkgs.texlive.withPackages (ps: with ps; [
        scheme-small
        latexmk
        # presentation / graphics
        beamer pgf pgfplots tcolorbox environ
        # language
        babel babel-spanish
        # math
        amsmath amsfonts amscls mathtools
        # layout / tables / misc
        geometry enumitem parskip caption float booktabs
        graphics hyperref url xcolor tools placeins
      ]);

      plotPython = pkgs.python3.withPackages (ps: with ps; [
        torch
        matplotlib
        ipython
      ]);

      # Build only the files needed by Proyecto/ slides (keeps the big
      # reference PDF in Referencias/ out of the store).
      proyectoSrc = lib.fileset.toSource {
        root = ./Proyecto;
        fileset = lib.fileset.fileFilter
          (f: builtins.elem f.name [
            "presentacion_beamer_funciones_bifurcacion_tent.tex"
            "logistic_map.png"
            "logistic_simple.py"
            "tent_map.png"
            "tent_function.png"
            "bifurcacion_simple.py"
          ])
          ./Proyecto;
      };

      proyecto-pdfs = pkgs.stdenvNoCC.mkDerivation {
        name = "proyecto-pdfs";
        src = proyectoSrc;
        nativeBuildInputs = [ tex ];
        # latexmk drives the multi-pass runs beamer needs (.nav, refs).
        buildPhase = ''
          runHook preBuild
          export HOME=$PWD
          for f in *.tex; do
            latexmk -pdf -interaction=nonstopmode -halt-on-error "$f"
          done
          runHook postBuild
        '';
        installPhase = ''
          runHook preInstall
          mkdir -p $out
          cp *.pdf $out/
          runHook postInstall
        '';
      };
    in {
      packages.${system} = {
        default = proyecto-pdfs;
        tex = tex;
        python = plotPython;
      };

      devShells.${system} = {
        default = pkgs.mkShell {
          packages = [ tex pkgs.poppler-utils ];
          shellHook = ''
            echo "LaTeX ready: pdflatex, latexmk, pdfinfo."
            echo "Live preview: latexmk -pdf -pvc Proyecto/presentacion_beamer_funciones_bifurcacion_tent.tex"
          '';
        };

        plots = pkgs.mkShell {
          packages = [ plotPython ];
          shellHook = ''
            echo "Plotting ready: python with torch and matplotlib."
          '';
        };
      };
    };
}
