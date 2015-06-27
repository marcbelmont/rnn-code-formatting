# RNN Code formatting #

This project lets you format source code using an LSTM-RNN that has been trained on a properly formatted piece of code. Before training the RNN, we encode the source code in a simplified format to facilitate the learning. Then the RNN is being fed the unformatted code and if the prediction does not match the next character of the input we consider that there is a formatting problem. We then explore the RNN predictions until a non-whitespace character that matches the next non-whitespace character of the input is found.

This approach has some nice advantages over standard formatting tools. You don't need to configure anything. The tool will **automatically learn** how many spaces you are using for indentation and the spacing around special characters.

The RNN and traning code is from  <https://github.com/karpathy/char-rnn> . I have only worked on formatting **CSS files** because that language is relatively simple. The tool will take care of code indentation and spaces between those characters ":;{}". Formatting other languages will require code changes.


Example:
--------

### Training file: ###
```css
body {
    font-family: sans-serif;
    font-size: 100%;
    background-color: #11303d;
    color: #000;
    margin: 0;
    padding: 0;
}

div.document {
    background-color: #1c4e63;
}

div.documentwrapper {
    float: left;
    width: 100%;
}
...
```

### Unformatted code: ###
```css
.foo {
  color: red;  background-color: blue;} #ac-main-content article.post div.entry p {
     color: #444;
                      font-size: 13px;
  text-align: left;
  padding: 0;
}
h5 { font-size:10px; color:#4062b7; font-weight:normal; padding-bottom:2px; }
h5 a { color:#4062b7; text-decoration:none; }
h5 a:hover { color:#4062b7; text-decoration:underline; }
```

### Formatted code: ###
```css
.foo {
    color: red;
    background-color: blue;
}

#ac-main-content article.post div.entry p {
    color: #444;
    font-size: 13px;
    text-align: left;
    padding: 0;
}

h5 {
    font-size: 10px;
    color: #4062b7;
    font-weight: normal;
    padding-bottom: 2px;
}

h5 a{
    color: #4062b7;
    text-decoration: none;
}

h5 a:hover {
    color: #4062b7;
    text-decoration: underline;
}
```

How to train the RNN:
---------------------
```
mkdir -p /tmp/rnn_train
cp test/formatted-1.css /tmp/rnn_train/input.txt
cd rnn; th train.lua -gpuid -1 -data_dir /tmp/rnn_train -eval_val_every 200 -checkpoint_dir /tmp/rnn_train/cv -print_every 10 -seq_length 12
```

How to format your source code:
-------------------------
```
cd rnn; th transform.lua -gpuid -1 /tmp/rnn_train/cv/your-model.t7 ../test/not-formatted-2.css /tmp/out.css
```
