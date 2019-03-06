@echo off
setlocal
%~d0
cd %~dp0
ipconfig
echo 正在以服务模式启动程序，此模式下不会自动开启浏览器，局域网内的设备都可以通过本机的ip地址来访问2333端口的页面
echo 警告：本程序没有安全设计，仅限于在安全的局域网内使用本模式
MoePhoto.exe -g