{
  description = "pixi env (FHS wrapped)";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };

        fhs = pkgs.buildFHSEnv {
          name = "pixi-env";

          targetPkgs = pkgs: [
            pkgs.pixi
            pkgs.git
            pkgs.cacert   
          ];
        };
      in {
        devShell = fhs.env;
      }
    );
}
