#panes.ps1
wt -p "PowerShell" -d . powershell -NoExit -File .\tail.ps1 spec `
   `; split-pane -H -p "PowerShell" -d . powershell -NoExit -File .\tail.ps1 coder `
   `; split-pane -V -p "PowerShell" -d . powershell -NoExit -File .\tail.ps1 judge `
   `; split-pane -H -p "PowerShell" -d . powershell -NoExit -File .\tail.ps1 tester
