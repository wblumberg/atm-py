import subprocess
import os
import nose_tests
print(__file__)
try:
    out = subprocess.check_output('nosetests-3.5 nose_tests.py', shell = True, stderr=subprocess.STDOUT)
    out = out.decode()
except subprocess.CalledProcessError as e:
    out = e.stdout.decode()

raus = open('testresults.log', 'w')
raus.write(out)
raus.close()

out_lines = out.splitlines()

title = 'atm-py unit test (%s)'%(out_lines[-1])
if out_lines[-1] == 'OK':
    message = out_lines[2]
else:
    message = out_lines[-6]

folder = os.path.split(nose_tests.__file__)[0]

openthis = folder + '/testresults.log'
os.system("terminal-notifier -title '%s' -message '%s' -execute 'open -a TextWrangler %s'"%(title,message,openthis))