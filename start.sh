#!/bin/bash

for i in `ps -ef | grep python | grep demo | awk {'print$2'}`
do
   echo Killing pid: "$i"
   kill -9 $i
done


nohup python ./demo-email.py > ./logs/myemaillog.out 2>&1 &
nohup python ./demo-chat.py > ./logs/mychat.out 2>&1 &

echo "starting demo chatbot ..."

sleep 10

echo "started demo chatbot"
