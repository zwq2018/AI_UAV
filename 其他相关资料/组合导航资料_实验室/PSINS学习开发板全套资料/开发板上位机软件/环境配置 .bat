cd /d %~dp0
copy cw3dgrph.ocx  C:\Windows\SysWOW64\
copy cwui.ocx      C:\Windows\SysWOW64\
copy cw3dgrph.ocx  C:\Windows\System32\
copy cwui.ocx      C:\Windows\System32\


regsvr32 C:\Windows\SysWOW64\cw3dgrph.ocx
regsvr32 C:\Windows\SysWOW64\cwui.ocx
regsvr32 C:\Windows\System32\cw3dgrph.ocx
regsvr32 C:\Windows\System32\cwui.ocx