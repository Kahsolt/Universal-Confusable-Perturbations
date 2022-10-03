@REM pgd
python attack.py -M resnet18 -D tiny-imagenet --method pgd --eps 0.1 --alpha 0.001

python attack.py -M resnet18 -D tiny-imagenet --method pgd --eps 0.03 --alpha 0.001 --alpha_to 0.0001
python attack.py -M resnet18 -D tiny-imagenet --method pgd --eps 0.03 --alpha 0.001 --alpha_to 0.00001

@REM mifgsm
python attack.py -M resnet18 -D tiny-imagenet --method mifgsm --eps 0.1 --alpha 0.001
python attack.py -M resnet18 -D tiny-imagenet --method mifgsm --eps 0.03 --alpha 0.001

@REM pgdl2
python attack.py -M resnet18 -D tiny-imagenet --method pgdl2 --eps 3 --alpha 0.01
python attack.py -M resnet18 -D tiny-imagenet --method pgdl2 --eps 1 --alpha 0.001
