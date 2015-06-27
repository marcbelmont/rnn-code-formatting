import hashlib
import unittest
import os
import subprocess


class Transform(unittest.TestCase):

    def setUp(self):
        os.chdir('rnn')

    def transform(self, model, path, timeout=False):
        command = 'luajit transform.lua -gpuid -1 %s %s /tmp/out.css' % (model, path)
        if timeout:
            command = "timeout %ss %s" % (timeout, command, )
        return subprocess.check_output(command.split())

    def test_transform(self):
        for model, path, expected in (
                ('../test/lm_lstm_epoch3.96_0.1381.t7',
                 '../test/not-formatted-1.css',
                 '15937ce04ee860b85ce1c0fdcc581be63059f19a'),
                ('../test/lm_lstm_epoch3.96_0.1381.t7',
                 '../test/not-formatted-2.css',
                 '5df190c246ae46504ea3cb78085ede87289c2aa3')):
            print self.transform(model, path, timeout=12)
            m = hashlib.sha1()
            with open('/tmp/out.css', 'r') as f:
                m.update(f.read())
            self.assertEqual(m.hexdigest(), expected)

if __name__ == '__main__':
    unittest.main()
