@REM pgd
python attack.py -M resnet18 -D svhn --method pgd --eps 0.1 --alpha 0.01
python attack.py -M resnet18 -D svhn --method pgd --eps 0.1 --alpha 0.005
python attack.py -M resnet18 -D svhn --method pgd --eps 0.1 --alpha 0.001
python attack.py -M resnet18 -D svhn --method pgd --eps 0.1 --alpha 0.0005

python attack.py -M resnet18 -D svhn --method pgd --eps 0.05 --alpha 0.01
python attack.py -M resnet18 -D svhn --method pgd --eps 0.05 --alpha 0.005
python attack.py -M resnet18 -D svhn --method pgd --eps 0.05 --alpha 0.001
python attack.py -M resnet18 -D svhn --method pgd --eps 0.05 --alpha 0.0005

python attack.py -M resnet18 -D svhn --method pgd --eps 0.03 --alpha 0.01
python attack.py -M resnet18 -D svhn --method pgd --eps 0.03 --alpha 0.005
python attack.py -M resnet18 -D svhn --method pgd --eps 0.03 --alpha 0.001
python attack.py -M resnet18 -D svhn --method pgd --eps 0.03 --alpha 0.0005

python attack.py -M resnet18 -D svhn --method pgd --eps 0.01 --alpha 0.01
python attack.py -M resnet18 -D svhn --method pgd --eps 0.01 --alpha 0.005
python attack.py -M resnet18 -D svhn --method pgd --eps 0.01 --alpha 0.001
python attack.py -M resnet18 -D svhn --method pgd --eps 0.01 --alpha 0.0005

@REM mifgsm
python attack.py -M resnet18 -D svhn --method mifgsm --eps 0.1 --alpha 0.01
python attack.py -M resnet18 -D svhn --method mifgsm --eps 0.1 --alpha 0.005
python attack.py -M resnet18 -D svhn --method mifgsm --eps 0.1 --alpha 0.001
python attack.py -M resnet18 -D svhn --method mifgsm --eps 0.1 --alpha 0.0005

python attack.py -M resnet18 -D svhn --method mifgsm --eps 0.05 --alpha 0.01
python attack.py -M resnet18 -D svhn --method mifgsm --eps 0.05 --alpha 0.005
python attack.py -M resnet18 -D svhn --method mifgsm --eps 0.05 --alpha 0.001
python attack.py -M resnet18 -D svhn --method mifgsm --eps 0.05 --alpha 0.0005

python attack.py -M resnet18 -D svhn --method mifgsm --eps 0.03 --alpha 0.01
python attack.py -M resnet18 -D svhn --method mifgsm --eps 0.03 --alpha 0.005
python attack.py -M resnet18 -D svhn --method mifgsm --eps 0.03 --alpha 0.001
python attack.py -M resnet18 -D svhn --method mifgsm --eps 0.03 --alpha 0.0005

python attack.py -M resnet18 -D svhn --method mifgsm --eps 0.01 --alpha 0.01
python attack.py -M resnet18 -D svhn --method mifgsm --eps 0.01 --alpha 0.005
python attack.py -M resnet18 -D svhn --method mifgsm --eps 0.01 --alpha 0.001
python attack.py -M resnet18 -D svhn --method mifgsm --eps 0.01 --alpha 0.0005

@REM pgdl2
python attack.py -M resnet18 -D svhn --method pgdl2 --eps 3 --alpha 0.01
python attack.py -M resnet18 -D svhn --method pgdl2 --eps 3 --alpha 0.005
python attack.py -M resnet18 -D svhn --method pgdl2 --eps 3 --alpha 0.001
python attack.py -M resnet18 -D svhn --method pgdl2 --eps 3 --alpha 0.0005

python attack.py -M resnet18 -D svhn --method pgdl2 --eps 1 --alpha 0.01
python attack.py -M resnet18 -D svhn --method pgdl2 --eps 1 --alpha 0.005
python attack.py -M resnet18 -D svhn --method pgdl2 --eps 1 --alpha 0.001
python attack.py -M resnet18 -D svhn --method pgdl2 --eps 1 --alpha 0.0005

python attack.py -M resnet18 -D svhn --method pgdl2 --eps 0.5 --alpha 0.01
python attack.py -M resnet18 -D svhn --method pgdl2 --eps 0.5 --alpha 0.005
python attack.py -M resnet18 -D svhn --method pgdl2 --eps 0.5 --alpha 0.001
python attack.py -M resnet18 -D svhn --method pgdl2 --eps 0.5 --alpha 0.0005

python attack.py -M resnet18 -D svhn --method pgdl2 --eps 0.3 --alpha 0.01
python attack.py -M resnet18 -D svhn --method pgdl2 --eps 0.3 --alpha 0.005
python attack.py -M resnet18 -D svhn --method pgdl2 --eps 0.3 --alpha 0.001
python attack.py -M resnet18 -D svhn --method pgdl2 --eps 0.3 --alpha 0.0005
