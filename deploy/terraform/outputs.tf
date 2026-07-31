output "droplet_ids" {
  description = "Map of env key -> droplet ID."
  value       = { for k, d in digitalocean_droplet.host : k => d.id }
}

output "droplet_names" {
  description = "Map of env key -> droplet name."
  value       = { for k, d in digitalocean_droplet.host : k => d.name }
}

output "droplet_ipv4" {
  description = "Map of env key -> public IPv4."
  value       = { for k, d in digitalocean_droplet.host : k => d.ipv4_address }
}

output "droplet_sizes" {
  description = "Map of env key -> size slug."
  value       = { for k, d in digitalocean_droplet.host : k => d.size }
}

output "firewall_id" {
  description = "Cloud firewall ID attached to both hosts."
  value       = digitalocean_firewall.base.id
}

output "firewall_name" {
  description = "Cloud firewall name."
  value       = digitalocean_firewall.base.name
}

output "region" {
  value = var.region
}

output "operator_ssh_sources" {
  description = "CIDRs allowed on port 22."
  value       = local.operator_ssh_sources
}