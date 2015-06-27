
-- misc utilities

function clone_list(tensor_list, zero_too)
    -- utility function. todo: move away to some utils file?
    -- takes a list of tensors and returns a list of cloned tensors
    local out = {}
    for k,v in pairs(tensor_list) do
        out[k] = v:clone()
        if zero_too then out[k]:zero() end
    end
    return out
end


function encode(text)
    local symbols = {}
    encoded = string.gsub(text, "([^ \n\t;:,={}()]+)", function (s)
                           table.insert(symbols, s)
                           return "x"
    end)
    return encoded, symbols
end

function decode(text, symbols)
    decoded = string.gsub(text, "[a-z]", function (s)
                           return table.remove(symbols, 1)
    end)
    return decoded
end
