
--[[

This file samples characters from a trained model

Code is based on implementation in
https://github.com/oxford-cs-ml-2015/practical6

]]--

require "torch"
require "nn"
require "nngraph"
require "optim"
require "lfs"

require "util.OneHot"
require "util.misc"

------------------------
-- Parse command line --
------------------------

cmd = torch.CmdLine()
cmd:text()
cmd:text("Sample from a character-level language model")
cmd:text()
cmd:text("Options")
-- required:
cmd:argument("-model","model checkpoint to use for sampling")
cmd:argument("-input", "input file")
cmd:argument("-output", "output file")
-- optional parameters
cmd:option("-gpuid",0,"which gpu to use. -1 = use CPU")
cmd:option("-min_prob",0.2,"only consider predictions above min_prob")
cmd:text()

-- parse input params
opt = cmd:parse(arg)

if opt.gpuid >= 0 then
    print("using CUDA on GPU " .. opt.gpuid .. "...")
    require "cutorch"
    require "cunn"
    cutorch.setDevice(opt.gpuid + 1) -- note +1 to make it 0 indexed! sigh lua
end

-------------------------------
-- load the model checkpoint --
-------------------------------

if not lfs.attributes(opt.model, "mode") then
    print("Error: File " .. opt.model .. " does not exist. Are you sure you didn't forget to prepend cv/ ?")
end
checkpoint = torch.load(opt.model)
protos = checkpoint.protos

-- initialize the vocabulary (and its inverted version)
local vocab = checkpoint.vocab
local ivocab = {}
for c,i in pairs(vocab) do ivocab[i] = c end

------------------------------
-- Initialize the rnn state --
------------------------------

local current_state = {}
local model = checkpoint.opt.model

local num_layers = checkpoint.opt.num_layers
for L=1,checkpoint.opt.num_layers do
    -- c and h for all layers
    local h_init = torch.zeros(1, checkpoint.opt.rnn_size)
    if opt.gpuid >= 0 then h_init = h_init:cuda() end
    table.insert(current_state, h_init:clone())
    table.insert(current_state, h_init:clone())
end
local state_size = #current_state

protos.rnn:evaluate() -- put in eval mode so that dropout works properly


-------------
-- Helpers --
-------------

function dir(obj)
    if obj == nil then
        return print(obj)
    end
    local result = {}
    for k, v in pairs(obj) do
        table.insert(result, k)
    end
    table.sort(result)
    print(table.concat(result, " "))
end

function print_table(t)
    local result = {}
    for _, v in ipairs(t) do
        table.insert(result, v)
    end
    print(table.concat(result, ","))
end

function print_tree(t, indentation)
    print_ws(indentation, t.score, t.completion)
    for _, v in ipairs(t.children) do
        print_tree(v, indentation .. "---")
    end
end


function print_ws(...)
    local result = {}
    for _, v in ipairs({...}) do
        if type(v) == "string" then
            v = v:gsub(" ", "S"):gsub("\n", "N")
        end
        if type(v) == "number" and v < 1 and v ~= 0 then
            v = string.format("%5.05f", v)
        end
        if v ~= nil then
            table.insert(result, v)
        end
    end
    print(table.concat(result, " "))
end

function tree_max(t, max)
    if max == nil then
        max = t
    end
    if t.score and t.score > max.score then
        max = t
    end
    for _, v in ipairs(t.children) do
        max = tree_max(v, max)
    end
    return max
end

function compare(a, b)
  return a[2] > b[2]
end

function show_stats(prediction)
    new ={}
    for k, v in pairs(torch.totable(prediction:squeeze())) do
        table.insert(new, {ivocab[k], v})
    end
    table.sort(new, compare)

    local stats = ""
    for _, x in pairs(new) do
        v = x[1] == "\n" and "N" or x[1]
        v = v == " " and "S" or v
        stats = stats .. string.format("%s %5.01f   ", v, x[2])
    end
    return stats
end

function copy_table(t)
    local result = {}
    for _, x  in ipairs(t) do
        table.insert(result, x:clone())
    end
    return result
end

function assign_table(t1, t2)
    for i, _ in ipairs(t1) do
        t1[i] = t2[i]
    end
end


function load_file(path)
    local f = torch.DiskFile(path)
    local rawdata = f:readString("*a")
    f:close()
    return rawdata
end

function next_non_space(text)
    for i=1, #text do
        local c = text:sub(i, i)
        if not is_space(c) then
            return c, i - 1
        end
    end
    return nil, nil
end

function is_space(c)
    return c == " " or c == "\n" or c == "\t"
end

---------
-- RNN --
---------

function matching_ns(current_state, input, predictions, j)
    local ns, pos = next_non_space(input:sub(j))
    if ns == nil then
        return 0, false
    end

    -- Explore predictions and pick the one with highest probability
    -- print_ws(input:sub(1,j-1), ns, j, pos)
    tree = explore_predictions(
        protos.rnn, current_state, predictions,
        "", 0, 1, ns)
    -- print_tree(tree, "+")

    -- pick best candidate
    best = tree_max(tree)
    assign_table(current_state, best.state)
    protos.rnn = best.rnn
    return pos, best.completion
end


function explore_predictions(
        rnn, state, predictions,
        completion, best, score, target)
    local node = nil
    -- Find the probability of the target
    for _, prediction in ipairs(predictions) do
        if target == prediction.c then
            local new_score = prediction.p * score
            if new_score > best then
                -- print_ws(best, new_score, completion)
                best = new_score
                node = {score=new_score, rnn=rnn, state=state, completion=completion, children={}}
            end
            break
        end
    end

    -- Stop exploring when the probability gets too low
    if not node then
        if score > opt.min_prob then
            node = {rnn=rnn, state=state, children={}}
        else
            return node
        end
    end

    -- Optimization: check if it's necessary to clone the RNN
    local branches = 0
    for _, prediction in ipairs(predictions) do
        if prediction.p * score > best and is_space(prediction.c) then
            branches = branches + 1
        end
    end

    -- Follow all promising whitespaces predictions
    local rnn_, state_
    for _, prediction in ipairs(predictions) do
        local new_score = prediction.p * score
        if new_score > best and is_space(prediction.c) then
            -- print_ws(new_score, completion, prediction.c)

            -- Check if cloning is needed
            if node.completion then
                rnn_ = rnn:clone()
                state_ = copy_table(state)
            elseif branches > 1 then
                rnn_ = rnn:clone() -- TODO can be optimized
                state_ = copy_table(state)
            else
                rnn_ = rnn
                state_ = state
            end

            -- Explore this branch
            local new_node = explore_predictions(
                rnn_,
                state_,
                rnn_next(rnn_, state_, prediction.c),
                completion .. prediction.c,
                best,
                new_score,
                target)
            if new_node then
                table.insert(node.children, new_node)
            end
            branches = branches - 1
        end
    end
    return node
end


function rnn_next_ns(rnn, current_state, c)
    local result = c
    while is_space(c) do
        c = rnn_next(rnn, current_state, c)[1].c
        result = result .. c
    end
    return result
end

function rnn_next(rnn, current_state, c)
    -- calculate prediction
    prev_char = torch.Tensor{vocab[c]}
    if opt.gpuid >= 0 then prev_char = prev_char:cuda() end
    local lst = rnn:forward{prev_char, unpack(current_state)}
    -- lst is a list of [state1,state2,..stateN,output]. We want everything but the last piece
    for i=1,state_size do
        current_state[i]= lst[i]
    end
    local predictions = lst[#lst] -- last element holds the log probabilities

    -- sort predictions
    local probabilities, indexes = torch.sort(predictions:squeeze(), 1, true)
    local result = {}
    probabilities = probabilities:apply(function(x) return torch.exp(x) end)
    for i, v in ipairs(torch.totable(indexes)) do
        table.insert(result, {c=ivocab[v], p=probabilities[i]})
    end
    return result
end

---------------
-- Transform --
---------------

local input, symbols = encode(load_file(opt.input))
local formatted = ""
local j = 1
local predictions

-- Feed the whole input to give the RNN some context
while j <= #input do
    rnn_next(protos.rnn, current_state, input:sub(j, j))
    j = j + 1
end

-- Reformat
j = 1
local c
while j <= #input do
    c = input:sub(j, j)
    formatted = formatted .. c
    predictions = rnn_next(protos.rnn, current_state, c)

    if input:sub(j + 1, j + 1) ~= predictions[1].c then
        local skip, completion  = matching_ns(current_state, input, predictions, j + 1)
        if completion then
            formatted = formatted .. completion:sub(1, #completion)
        end
        j = j + skip
    end

    j = j + 1
end

file = io.open(opt.output, "w")
io.output(file)
io.write(decode(formatted, symbols))
io.close(file)
