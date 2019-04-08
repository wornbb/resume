#!/usr/bin/zsh
while [ ! -f ./Yaswanth2c/v11 ]
do
  sleep 600
done
echo "Subject: Core2 completed\n\nHi Yi, the core 2 files are ready" | ssmtp marco.kassis@mail.mcgill.ca 
~
while [ ! -f ./Yaswanth4c/v11 ]
do
  sleep 600
done
echo "Subject: Core4 completed\n\nHi Yi, the core 4 files are ready" | ssmtp marco.kassis@mail.mcgill.ca 
while [ ! -f ./Yaswanth8c/v11 ]
do
  sleep 600
done
echo "Subject: Core8 completed\n\nHi Yi, the core 8 files are ready" | ssmtp marco.kassis@mail.mcgill.ca 
while [ ! -f ./Yaswanth16c/v11 ]
do
  sleep 600
done
echo "Subject: Core16 completed\n\nHi Yi, the core 16 files are ready" | ssmtp marco.kassis@mail.mcgill.ca 
~
