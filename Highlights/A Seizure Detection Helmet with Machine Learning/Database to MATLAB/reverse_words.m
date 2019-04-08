function strOut = reverse_words(strIn)
    strOut = strtrim(strIn);
    if ~isempty(strOut)
        % Could use strsplit() instead of textscan() in R2013a or later
        words = textscan(strOut, '%s');
        words = words{1};
        strOut = strtrim(sprintf('%s ', words{end:-1:1}));
    end
end