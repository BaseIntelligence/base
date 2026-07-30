terraform {
  required_version = ">= 1.5.0"

  required_providers {
    digitalocean = {
      source  = "digitalocean/digitalocean"
      version = "~> 2.50"
    }
  }

  # Local state by default. Prefer a remote backend (DO Spaces / S3) in
  # production so state is not on a laptop. Never commit *.tfstate*.
}