#!/bin/sh

# sudo apt-get install tigervnc-viewer

if ! [ -f ~/.vnc/mersewm_passwd ]; then
  echo "passwd File already exists, skipping download"
  scp travis@travis-wm:/home/travis/.vnc/passwd ~/.vnc/mersewm_passwd
fi

echo "opening ssh tunnel"
ssh -fL 6001:localhost:5901 travis@travis-WM sleep 1

echo "connecting vnc viewer"
vncviewer -SecurityTypes VncAuth -passwd ~/.vnc/mersewm_passwd localhost:6001 
