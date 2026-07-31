locals {
  base_tags = distinct(concat(
    [
      var.project_name,
      "managed-by:terraform",
      "cost-center:base",
    ],
    var.tags,
  ))

  # Normalize operator sources to CIDR form for the firewall API.
  operator_ssh_sources = [
    for c in var.operator_ipv4_cidrs :
    can(regex("/", c)) ? c : "${c}/32"
  ]
}

resource "digitalocean_droplet" "host" {
  for_each = var.environments

  name   = each.value.name
  region = var.region
  size   = var.droplet_size
  image  = var.image

  ssh_keys = var.ssh_key_ids

  # Monitoring + IPv6; private networking helps later VPC work.
  monitoring = true
  ipv6       = true
  vpc_uuid   = null

  tags = distinct(concat(local.base_tags, [
    "env:${each.key}",
    "role:base-host",
  ]))

  user_data = templatefile("${path.module}/cloud-init.yaml.tftpl", {
    env_name = each.key
  })

  lifecycle {
    ignore_changes = [
      # Avoid recreate if DO rewrites user_data whitespace on read-back.
      user_data,
    ]
  }
}

resource "digitalocean_firewall" "base" {
  name = "${var.project_name}-hosts"

  droplet_ids = [for d in digitalocean_droplet.host : d.id]

  tags = local.base_tags

  # SSH: operator IP only (never 0.0.0.0/0).
  inbound_rule {
    protocol         = "tcp"
    port_range       = "22"
    source_addresses = local.operator_ssh_sources
  }

  # HTTP / HTTPS open to the world (TLS terminates on-box later).
  inbound_rule {
    protocol         = "tcp"
    port_range       = "80"
    source_addresses = ["0.0.0.0/0", "::/0"]
  }

  inbound_rule {
    protocol         = "tcp"
    port_range       = "443"
    source_addresses = ["0.0.0.0/0", "::/0"]
  }

  # ICMP for diagnostics.
  inbound_rule {
    protocol         = "icmp"
    source_addresses = ["0.0.0.0/0", "::/0"]
  }

  outbound_rule {
    protocol              = "tcp"
    port_range            = "1-65535"
    destination_addresses = ["0.0.0.0/0", "::/0"]
  }

  outbound_rule {
    protocol              = "udp"
    port_range            = "1-65535"
    destination_addresses = ["0.0.0.0/0", "::/0"]
  }

  outbound_rule {
    protocol              = "icmp"
    destination_addresses = ["0.0.0.0/0", "::/0"]
  }
}