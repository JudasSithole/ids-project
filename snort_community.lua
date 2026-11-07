-- Snort 3 community-rules offline evaluation config (Friday CICIDS2017)
HOME_NET = 'any'
EXTERNAL_NET = 'any'

dofile('/home/judas/ids-project/snort_defaults.lua')

stream = { }
stream_ip = { }
stream_icmp = { }
stream_tcp = { }
stream_udp = { }
stream_user = { }
stream_file = { }

arp_spoof = { }
back_orifice = { }
dns = { }
imap = { }
netflow = {}
normalizer = { }
pop = { }
rpc_decode = { }
sip = { }
smtp = { }
ssh = { }
ssl = { }
telnet = { }

dce_smb = { }
dce_tcp = { }
dce_udp = { }
dce_http_proxy = { }
dce_http_server = { }

http_inspect = { }
http2_inspect = { }

file_id = { rules_file = '/home/judas/ids-project/file_magic.rules' }
file_policy = { }
js_norm = default_js_norm

appid = { }
wizard = {
  spells = {
    { service = 'http', proto = 'tcp', to_server = {'GET', 'POST'}, to_client = {'HTTP/'} },
  }
}
binder = {
  { when = { proto = 'udp', ports = '53' }, use = { type = 'dns' } },
  { when = { service = 'dns' },              use = { type = 'dns' } },
  { when = { proto = 'tcp', ports = '111' }, use = { type = 'rpc_decode' } },
  { when = { proto = 'tcp', ports = '80 8080' }, use = { type = 'http_inspect' } },
  { when = { service = 'http' },             use = { type = 'http_inspect' } },
  { when = { proto = 'tcp', ports = '25' },  use = { type = 'smtp' } },
  { use = { type = 'wizard' } }
}

ips = {
  enable_builtin_rules = true,
  variables = default_variables,
  include = '/home/judas/ids-project/snort_rules/snort3-community-rules/snort3-community.rules'
}

alert_fast = { file = true }
