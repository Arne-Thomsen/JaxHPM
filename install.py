#!/usr/bin/env python3
"""
Custom installation script for JaxHPM that conditionally installs JAX.
Usage: python install.py
"""
import subprocess
import sys


def is_jax_installed():
    """Check if JAX is already installed."""
    try:
        import jax

        print(f"✓ JAX is already installed (version {jax.__version__})")
        return True
    except ImportError:
        print("✗ JAX is not installed")
        return False


def main():
    """Install JaxHPM with conditional JAX installation."""
    print("Installing JaxHPM...")

    # Install the package without JAX first
    cmd = [sys.executable, "-m", "pip", "install", "-e", "."]

    if not is_jax_installed():
        print("\nInstalling JAX as an optional dependency...")
        cmd.append("[jax]")
    else:
        print("\nSkipping JAX installation (already available)")

    print(f"\nRunning: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False)

    if result.returncode == 0:
        print("\n✓ JaxHPM installation completed successfully!")
    else:
        print("\n✗ Installation failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
