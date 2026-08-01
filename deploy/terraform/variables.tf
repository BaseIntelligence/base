variable "project_name" {
  type        = string
  description = "Prefix for resource names and cost tags."
  default     = "base"
}

variable "region" {
  type        = string
  description = "DigitalOcean region slug."
  default     = "nyc1"
}

variable "droplet_size" {
  type = string
  # Plan originally named s-8vcpu-16gb ($96). That slug is no longer offered;
  # s-8vcpu-16gb-amd is the current 8 vCPU / 16 GB Basic AMD equivalent.
  description = "Droplet size slug (8 vCPU / 16 GB)."
  default     = "s-8vcpu-16gb-amd"
}

variable "image" {
  type        = string
  description = "Base image slug."
  default     = "ubuntu-24-04-x64"
}

variable "ssh_key_ids" {
  type        = list(number)
  description = "DigitalOcean SSH key IDs injected at create time."
}

variable "operator_ipv4_cidrs" {
  type        = list(string)
  description = "CIDRs allowed to reach SSH (port 22). Typically a single /32."
}

variable "tags" {
  type        = list(string)
  description = "Extra tags applied to droplets and firewall (cost tracking)."
  default     = []
}

variable "environments" {
  type = map(object({
    name = string
  }))
  description = "Map of environment key -> droplet display name."
  default = {
    staging           = { name = "gbase-staging" }
    staging_validator = { name = "gbase-staging-validator" }
    prod              = { name = "gbase-prod" }
    prod_validator    = { name = "gbase-prod-validator" }
  }
}