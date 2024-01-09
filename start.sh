#!/bin/bash

for i in `ps -ef | grep python | grep demo | awk {'print$2'}`
do
   echo Killing pid: "$i"
   kill -9 $i
done


nohup python ./demo-email-en.py > ./logs/emaillog-en.out 2>&1 &
nohup python ./demo-email-zh.py > ./logs/emaillog-zh.out 2>&1 &
nohup python ./demo-chat-en.py > ./logs/chatlog-en.out 2>&1 &
nohup python ./demo-chat-zh.py > ./logs/chatlog-zh.out 2>&1 &

echo "starting demo chatbot ..."

sleep 10

echo "started demo chatbot"
