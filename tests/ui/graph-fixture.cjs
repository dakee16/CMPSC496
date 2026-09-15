// Synthetic nested-loop example from the reported dictionary-inversion issue.
module.exports={
  nodes:[["s","start","Start"],["a","step","unique_values = []"],["b","loop","For v in values"],
    ["c","step","count = 0"],["d","loop","For other in values"],["e","branch","If other == v"],
    ["f","step","count += 1"],["g","branch","If count == 1"],["h","step","unique_values.append(v)"],
    ["i","step","inverted = {}"],["j","loop","For i in range(len(keys))"],
    ["k","branch","If values[i] in unique_values"],["l","step","inverted[values[i]] = keys[i]"],
    ["m","return","return inverted"]].map(([id,kind,label])=>({id,kind,label})),
  edges:[["s","a"],["a","b"],["b","c","repeat"],["c","d"],["d","e","repeat"],
    ["e","f","yes"],["f","d","repeat"],["e","d","no"],["d","g","done"],["g","h","yes"],
    ["h","b","repeat"],["g","b","no"],["b","i","done"],["i","j"],["j","k","repeat"],
    ["k","l","yes"],["l","j","repeat"],["k","j","no"],["j","m","done"]].map(([src,dst,label])=>({src,dst,label}))
};
