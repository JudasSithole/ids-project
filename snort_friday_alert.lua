HOME_NET = 'any'
EXTERNAL_NET = 'any'
dofile('/usr/local/etc/snort/snort_defaults.lua')
dofile('/usr/local/etc/snort/snort.lua')
alert_fast = { file = true }
