### Manual startup 
Using Elasticsearch 5.5 image

```sh
sudo bash
service stop elasticsearch
yum update elasticsearch-5.6.11
elasticsearch-plugin remove repository-s3
elasticsearch-plugin install repository-s3
chown -R elasticsearch /etc/elasticsearch
mkfs -t ext4 /dev/nvme0n1
mount /dev/nvme0n1 /mnt/elasticsearch_nvme1
mkdir /mnt/elasticsearch_nvme1/nodes
chown -R elasticsearch /mnt/elasticsearch_nvme1
chgrp -R elasticsearch /mnt/elasticsearch_nvme1

# Edit /etc/elasticsearch/elasticsearch.yml config to include new discovery.zen.ping.unicast.hosts
# This should use the internal IP (external should not be generally accessible)
# We should automate this...have had trouble with the aws plugin, at least at this version

# Edit web server .env and all compute nodes bystro/elastic-config/config.yml, adding in the internal (vpc-private) ips of all elasticsearch nodes.
# In the future we should redploy the relevant pods

service start elasticsearch
```
