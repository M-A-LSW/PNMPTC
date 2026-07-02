function output = soft_thresholding(input, threshold)
    level = 4;
    [C,S]=wavedec2(input, level, 'db4');
    out = sign(C) .* max(abs(C) - threshold, 0);
    output = waverec2(out, S, 'db4');
end