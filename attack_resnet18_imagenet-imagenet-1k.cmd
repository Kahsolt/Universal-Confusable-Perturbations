@REM pgd
python attack.py -M resnet18 -D imagenet-1k --method pgd --eps 0.03 --alpha 0.001
python attack.py -M resnet18 -D imagenet-1k --method pgd --eps 0.03 --alpha 0.0001
python attack.py -M resnet18 -D imagenet-1k --method pgd --eps 0.03 --alpha 0.00001

rem python attack.py -M resnet18 -D imagenet-1k --method pgd --eps 0.03 --alpha 0.001 --alpha_to 0.0001
rem python attack.py -M resnet18 -D imagenet-1k --method pgd --eps 0.03 --alpha 0.001 --alpha_to 0.00001

@REM mifgsm
python attack.py -M resnet18 -D imagenet-1k --method mifgsm --eps 0.03 --alpha 0.001

@REM pgdl2
python attack.py -M resnet18 -D imagenet-1k --method pgdl2 --eps 1 --alpha 0.001