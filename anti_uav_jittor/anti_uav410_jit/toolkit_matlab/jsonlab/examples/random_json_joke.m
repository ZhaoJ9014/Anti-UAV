function jokes=random_json_joke(num, url)
% this example shows how to use the _DataLink_ annotation defined in the
% JData specification
% (https://github.com/NeuroJSON/jdata/blob/master/JData_specification.md#data-referencing-and-links)
% to define linked JSON/binary JSON data using external files or URL on the
% web. In the below example, the jokeapi.dev feed returns a JSON record via
% RESTFul URL, the returned record contains a subfield called `joke`, which
% can be retrieved via the JSONPath $.joke attched after the URL, separated
% by a colon. The general _DataLink_ URL is in the form of "URL:$jsonpath"

if(nargin==0)
    num=1;
end

if(nargin<2)
    url='https://v2.jokeapi.dev/joke/Programming?type=single';
end

joke.(encodevarname('_DataLink_'))=[url ':$.joke'];
jurl=savejson('',joke);

jokes=cell(1,num);
for i=1:num
    jokes{i}=loadjson(jurl, 'maxlinklevel',1);
end

if(num==1)
    jokes=jokes{1};
end