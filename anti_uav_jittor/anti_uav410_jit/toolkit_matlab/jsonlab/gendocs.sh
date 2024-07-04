#!/bin/sh

#============================================================
#  JSONLAB inline documentation to wiki convertor
#
#  Author: Qianqian Fang <q.fang at neu.edu>
#============================================================

print_help()
{
   awk '/^%/ {dp=1} / this file is part of EasyH5/ {exit} \
        /-- this function is part of JSONLab/ {exit} \
        /^function/ {dp=1} /./ {if(dp==1) print;}' $1 \
     | grep -v 'Qianqian' | grep -v 'date:' | #grep -v '^%\s*$'| \
     sed -e 's/^function\(.*$\)/\n%==== function\1 ====/g'
}
print_group()
{
   for fun in $@
   do 
      print_help $fun.m
   done
   echo ''
}

func_jdata="jdataencode jdatadecode"
func_json="loadjson savejson"
func_bjdata="loadbj savebj"
func_ubjson="loadubjson saveubjson"
func_msgpack="loadmsgpack savemsgpack"
func_space="jsave jload"
func_interface="loadjd savejd"
func_mmap="jsonget jsonset getfromjsonpath filterjsonmmap"
func_zip="zlibencode zlibdecode gzipencode gzipdecode lzmaencode lzmadecode
          lzipencode lzipdecode lz4encode lz4decode lz4hcencode lz4hcdecode
	  base64encode base64decode encodevarname decodevarname"
func_helper="jsonopt mergestruct varargin2struct match_bracket
          fast_match_bracket nestbracket2dim"

echo %%=== "#" JData specification ===
print_group $func_jdata

echo %%=== "#" JSON ===
print_group $func_json

echo %%=== "#" BJData ===
print_group $func_bjdata

echo %%=== "#" UBJSON ===
print_group $func_ubjson

echo %%=== "#" MessagePack ===
print_group $func_msgpack

echo %%=== "#" Workspace ===
print_group $func_space

echo %%=== "#" Interface ===
print_group $func_interface

echo %%=== "#" Memory-map ===
print_group $func_mmap

echo %%=== "#" Compression and decompression ===
print_group $func_zip

echo %%=== "#" Miscellaneous functions ===
print_group $func_helper

