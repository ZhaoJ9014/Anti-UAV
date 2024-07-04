function run_jsonlab_test(tests)
%
% run_jsonlab_test
%   or
% run_jsonlab_test(tests)
% run_jsonlab_test({'js','jso','bj','bjo'})
%
% Unit testing for JSONLab JSON, BJData/UBJSON encoders and decoders
%
% authors:Qianqian Fang (q.fang <at> neu.edu)
% date: 2020/06/08
%
% input:
%      tests: is a cell array of strings, possible elements include
%         'js':  test savejson/loadjson
%         'jso': test savejson/loadjson special options
%         'bj':  test savebj/loadbj
%         'bjo': test savebj/loadbj special options
%
% license:
%     BSD or GPL version 3, see LICENSE_{BSD,GPLv3}.txt files for details 
%
% -- this function is part of JSONLab toolbox (http://iso2mesh.sf.net/cgi-bin/index.cgi?jsonlab)
%

if(nargin==0)
    tests={'js','jso','bj','bjo','jmap','bmap','bugs'};
end

%%
if(ismember('js',tests))
    fprintf(sprintf('%s\n',char(ones(1,79)*61)));
    fprintf('Test JSON functions\n');
    fprintf(sprintf('%s\n',char(ones(1,79)*61)));

    test_jsonlab('single integer',@savejson,5,'[5]');
    test_jsonlab('single float',@savejson,3.14,'[3.14]');
    test_jsonlab('nan',@savejson,nan,'["_NaN_"]');
    test_jsonlab('inf',@savejson,inf,'["_Inf_"]');
    test_jsonlab('-inf',@savejson,-inf,'["-_Inf_"]');
    test_jsonlab('large integer',@savejson,uint64(2^64),'[18446744073709551616]');
    test_jsonlab('large negative integer',@savejson,int64(-2^63),'[-9223372036854775808]');
    test_jsonlab('boolean as 01',@savejson,[true,false],'[1,0]','compact',1);
    test_jsonlab('empty array',@savejson,[],'[]');
    test_jsonlab('empty cell',@savejson,{},'[]');
    test_jsonlab('empty struct',@savejson,struct,'{}','compact',1);
    test_jsonlab('empty struct with fields',@savejson,repmat(struct('a',1),0,1),'[]');
    test_jsonlab('empty string',@savejson,'','""','compact',1);
    test_jsonlab('string escape',@savejson,sprintf('jdata\n\b\ashall\tprevail\t"\"\\'),'"jdata\n\b\ashall\tprevail\t\"\"\\"');
    if(exist('string'))
        test_jsonlab('string type',@savejson,string(sprintf('jdata\n\b\ashall\tprevail')),'["jdata\n\b\ashall\tprevail"]','compact',1);
        test_jsonlab('string array',@savejson,[string('jdata'),string('shall'),string('prevail')],'["jdata","shall","prevail"]','compact',1);
    end
    test_jsonlab('empty name',@savejson,loadjson('{"":""}'),'{"":""}','compact',1);
    if(exist('containers.Map'))
        test_jsonlab('empty name with map',@savejson,loadjson('{"":""}','usemap',1),'{"":""}','compact',1);
    end
    test_jsonlab('row vector',@savejson,[1,2,3],'[1,2,3]');
    test_jsonlab('column vector',@savejson,[1;2;3],'[[1],[2],[3]]','compact',1);
    test_jsonlab('mixed array',@savejson,{'a',1,0.9},'["a",1,0.9]','compact',1);
    test_jsonlab('mixed array from string',@savejson,loadjson('["a",{"c":1}, [2,3]]'),'["a",{"c":1},[2,3]]','compact',1);
    test_jsonlab('char array',@savejson,['AC';'EG'],'["AC","EG"]','compact',1);
    test_jsonlab('maps',@savejson,struct('a',1,'b','test'),'{"a":1,"b":"test"}','compact',1);
    test_jsonlab('2d array',@savejson,[1,2,3;4,5,6],'[[1,2,3],[4,5,6]]','compact',1);
    test_jsonlab('non-uniform 2d array',@savejson,{[1,2],[3,4,5],[6,7]},'[[1,2],[3,4,5],[6,7]]','compact',1);
    test_jsonlab('non-uniform array with length multiple of first element',@savejson,{[1,2],[3,4,5,6],[7,8]},'[[1,2],[3,4,5,6],[7,8]]','compact',1);
    test_jsonlab('1d array with flexible white space',@savejson,loadjson(sprintf(' [ +1, \n -2e3 \n , 3.0E+00 ,\r+4e-0] ')),'[1,-2000,3,4]','compact',1);
    test_jsonlab('2d array with flexible white space',@savejson,loadjson(sprintf(' [\r [\n 1 , \r\n  2\n, 3] ,\n[ 4, 5 , \t 6\t]\n] ')),'[[1,2,3],[4,5,6]]','compact',1);
    test_jsonlab('3d (row-major) nested array',@savejson,reshape(1:(2*3*2),2,3,2),...
         '[[[1,7],[3,9],[5,11]],[[2,8],[4,10],[6,12]]]','compact',1,'nestarray',1);
    test_jsonlab('3d (column-major) nested array',@savejson,reshape(1:(2*3*2),2,3,2),...
         '[[[1,2],[3,4],[5,6]],[[7,8],[9,10],[11,12]]]','compact',1,'nestarray',1,'formatversion',1.9);
    test_jsonlab('3d annotated array',@savejson,reshape(int8(1:(2*3*2)),2,3,2),...
         '{"_ArrayType_":"int8","_ArraySize_":[2,3,2],"_ArrayData_":[1,7,3,9,5,11,2,8,4,10,6,12]}','compact',1);
    test_jsonlab('complex number',@savejson,single(2+4i),...
         '{"_ArrayType_":"single","_ArraySize_":[1,1],"_ArrayIsComplex_":true,"_ArrayData_":[[2],[4]]}','compact',1);
    test_jsonlab('empty sparse matrix',@savejson,sparse(2,3),...
         '{"_ArrayType_":"double","_ArraySize_":[2,3],"_ArrayIsSparse_":true,"_ArrayData_":[]}','compact',1);
    test_jsonlab('real sparse matrix',@savejson,sparse([0,3,0,1,4]'),...
         '{"_ArrayType_":"double","_ArraySize_":[5,1],"_ArrayIsSparse_":true,"_ArrayData_":[[2,4,5],[3,1,4]]}','compact',1);
    test_jsonlab('complex sparse matrix',@savejson,sparse([0,3i,0,1,4i].'),...
         '{"_ArrayType_":"double","_ArraySize_":[5,1],"_ArrayIsComplex_":true,"_ArrayIsSparse_":true,"_ArrayData_":[[2,4,5],[0,1,0],[3,0,4]]}','compact',1);
    test_jsonlab('heterogeneous cell',@savejson,{{1,{2,3}},{4,5},{6};{7},{8,9},{10}},...
         '[[[1,[2,3]],[4,5],[6]],[[7],[8,9],[10]]]','compact',1);
    test_jsonlab('struct array',@savejson,repmat(struct('i',1.1,'d','str'),[1,2]),...
         '[{"i":1.1,"d":"str"},{"i":1.1,"d":"str"}]','compact',1);
    test_jsonlab('encoded fieldnames',@savejson,struct(encodevarname('_i'),1,encodevarname('i_'),'str'),...
         '{"_i":1,"i_":"str"}','compact',1);
    if(exist('OCTAVE_VERSION','builtin')~=0)
        test_jsonlab('encoded fieldnames without decoding',@savejson,struct(encodevarname('_i'),1,encodevarname('i_'),'str'),...
             '{"_i":1,"i_":"str"}','compact',1,'UnpackHex',0);
    else
        test_jsonlab('encoded fieldnames without decoding',@savejson,struct(encodevarname('_i'),1,encodevarname('i_'),'str'),...
             '{"x0x5F_i":1,"i_":"str"}','compact',1,'UnpackHex',0);
    end       
    if(exist('containers.Map'))
         test_jsonlab('containers.Map',@savejson,containers.Map({'Andy','^_^'},{true,'-_-'}),...
             '{"Andy":true,"^_^":"-_-"}','compact',1,'usemap',1);
    end
    if(exist('istable'))
         test_jsonlab('simple table',@savejson,table({'Andy','^_^'},{true,'-_-'}),...
             '{"_TableCols_":["Var1","Var2"],"_TableRows_":[],"_TableRecords_":[["Andy","^_^"],[true,"-_-"]]}','compact',1);
    end
    if(exist('bandwidth'))
         lband=2;
         uband=1;
         a=double(full(spdiags(true(4,lband+uband+1),-uband:lband,3,4)));
         a(a~=0)=find(a);

         test_jsonlab('lower band matrix',@savejson,tril(a),...
             '{"_ArrayType_":"double","_ArraySize_":[3,4],"_ArrayZipSize_":[2,3],"_ArrayShape_":["lowerband",1],"_ArrayData_":[1,5,9,0,2,6]}','compact',1,'usearrayshape',1);
         test_jsonlab('upper band matrix',@savejson,triu(a),...
             '{"_ArrayType_":"double","_ArraySize_":[3,4],"_ArrayZipSize_":[3,3],"_ArrayShape_":["upperband",2],"_ArrayData_":[7,11,0,4,8,12,1,5,9]}','compact',1,'usearrayshape',1);
         test_jsonlab('diag matrix',@savejson,tril(triu(a)),...
             '{"_ArrayType_":"double","_ArraySize_":[3,4],"_ArrayShape_":"diag","_ArrayData_":[1,5,9]}','compact',1,'usearrayshape',1);
         test_jsonlab('band matrix',@savejson,a,...
             '{"_ArrayType_":"double","_ArraySize_":[3,4],"_ArrayZipSize_":[4,3],"_ArrayShape_":["band",2,1],"_ArrayData_":[7,11,0,4,8,12,1,5,9,0,2,6]}','compact',1,'usearrayshape',1);
         a=a(:,1:3);
         a=uint8(tril(a)+tril(a)');
         test_jsonlab('symmetric band matrix',@savejson,a,...
             '{"_ArrayType_":"uint8","_ArraySize_":[3,3],"_ArrayZipSize_":[2,3],"_ArrayShape_":["lowersymmband",1],"_ArrayData_":[2,10,18,0,2,6]}','compact',1,'usearrayshape',1);
         a(a==0)=1;
         test_jsonlab('lower triangular matrix',@savejson,tril(a),...
             '{"_ArrayType_":"uint8","_ArraySize_":[3,3],"_ArrayShape_":"lower","_ArrayData_":[2,2,10,1,6,18]}','compact',1,'usearrayshape',1);
         test_jsonlab('upper triangular matrix',@savejson,triu(a),...
             '{"_ArrayType_":"uint8","_ArraySize_":[3,3],"_ArrayShape_":"upper","_ArrayData_":[2,2,1,10,6,18]}','compact',1,'usearrayshape',1);
    end
    try
        val=zlibencode('test');
        a=uint8(eye(5));
        a(20,1)=1;
        test_jsonlab('zlib/zip compression (level 6)',@savejson,a,...
            sprintf('{"_ArrayType_":"uint8","_ArraySize_":[20,5],"_ArrayZipSize_":[1,100],"_ArrayZipType_":"zlib","_ArrayZipData_":"eJxjZAABRhwkxQBsDAACIQAH\n"}'),...
            'compact',1, 'Compression','zlib','CompressArraySize',0)  % nestarray for 4-D or above is not working
        test_jsonlab('gzip compression (level 6)',@savejson,a,...
            sprintf('{"_ArrayType_":"uint8","_ArraySize_":[20,5],"_ArrayZipSize_":[1,100],"_ArrayZipType_":"gzip","_ArrayZipData_":"H4sIAAAAAAAAA2NkAAFGHCTFAGwMAF9Xq6VkAAAA\n"}'),...
            'compact',1, 'Compression','gzip','CompressArraySize',0)  % nestarray for 4-D or above is not working
        test_jsonlab('lzma compression (level 5)',@savejson,a,...
            sprintf('{"_ArrayType_":"uint8","_ArraySize_":[20,5],"_ArrayZipSize_":[1,100],"_ArrayZipType_":"lzma","_ArrayZipData_":"XQAAEABkAAAAAAAAAAAAgD1IirvlZSEY7DH///taoAA=\n"}'),...
            'compact',1, 'Compression','lzma','CompressArraySize',0)  % nestarray for 4-D or above is not working
    catch
    end
end
%%
if(ismember('jso',tests))
    fprintf(sprintf('%s\n',char(ones(1,79)*61)));
    fprintf('Test JSON function options\n');
    fprintf(sprintf('%s\n',char(ones(1,79)*61)));

    test_jsonlab('boolean',@savejson,[true,false],'[true,false]','compact',1,'ParseLogical',1);
    test_jsonlab('nan option',@savejson,nan,'["_nan_"]','NaN','"_nan_"');
    test_jsonlab('inf option',@savejson,-inf,'["-inf"]','Inf','"$1inf"');
    test_jsonlab('output int format',@savejson,uint8(5),'[  5]','IntFormat','%3d');
    test_jsonlab('output float format',@savejson,pi,'[3.142]','FloatFormat','%5.3f');
    test_jsonlab('remove singlet array',@savejson,{struct('a',1),5},'[{"a":1},5]','compact',1,'SingletArray',0);
    test_jsonlab('keep singlet array',@savejson,{struct('a',1),5},'[[{"a":[1]}],[5]]','compact',1,'SingletArray',1);
    test_jsonlab('test no datalink',@savejson,loadjson(savejson('a',struct(encodevarname('_DataLink_'),...
            '../examples/example2.json:$.glossary.title'))),'{"a":[{"_DataLink_":"../examples/example2.json:$.glossary.title"}]}','compact',1,'SingletArray',1);
    test_jsonlab('test maxlinklevel',@savejson,loadjson(savejson('a',struct(encodevarname('_DataLink_'),...
            '../examples/example2.json:$.glossary.title')),'maxlinklevel',1),'{"a":"example glossary"}','compact',1,'SingletArray',1);
end


%%
if(ismember('bj',tests))
    fprintf(sprintf('%s\n',char(ones(1,79)*61)));
    fprintf('Test Binary JSON functions\n');
    fprintf(sprintf('%s\n',char(ones(1,79)*61)));

    test_jsonlab('uint8 integer',@savebj,2^8-1,'U<255>','debug',1);
    test_jsonlab('uint16 integer',@savebj,2^8,'u<256>','debug',1);
    test_jsonlab('int8 integer',@savebj,-2^7,'i<-128>','debug',1);
    test_jsonlab('int16 integer',@savebj,-2^7-1,'I<-129>','debug',1);
    test_jsonlab('int32 integer',@savebj,-2^15-1,'l<-32769>','debug',1);
    test_jsonlab('uint16 integer',@savebj,2^16-1,'u<65535>','debug',1);
    test_jsonlab('uint32 integer',@savebj,2^16,'m<65536>','debug',1);
    test_jsonlab('uint32 integer',@savebj,2^32-1,'m<4294967295>','debug',1);
    test_jsonlab('int32 integer',@savebj,-2^31,'l<-2147483648>','debug',1);
    test_jsonlab('single float',@savebj,3.14,'D<3.14>','debug',1);
    test_jsonlab('nan',@savebj,nan,'D<NaN>','debug',1);
    test_jsonlab('inf',@savebj,inf,'D<Inf>','debug',1);
    test_jsonlab('-inf',@savebj,-inf,'D<-Inf>','debug',1);
    test_jsonlab('uint64 integer',@savebj,uint64(2^64),'M<18446744073709551616>','debug',1);
    test_jsonlab('int64 negative integer',@savebj,int64(-2^63),'L<-9223372036854775808>','debug',1);
    test_jsonlab('boolean as 01',@savebj,[true,false],'[U<1>U<0>]','debug',1,'nestarray',1);
    test_jsonlab('empty array',@savebj,[],'Z','debug',1);
    test_jsonlab('empty cell',@savebj,{},'Z','debug',1);
    test_jsonlab('empty string',@savebj,'','SU<0>','debug',1);
    test_jsonlab('skip no-op before marker and after value',@savebj,loadbj(char(['NN[NU' char(5) 'NNNU' char(1) ']'])),'[$U#U<2><5><1>','debug',1);
    test_jsonlab('string escape',@savebj,sprintf('jdata\n\b\ashall\tprevail\t"\"\\'),sprintf('SU<25>jdata\n\b\ashall\tprevail\t\"\"\\'),'debug',1);
    if(exist('string'))
        test_jsonlab('string type',@savebj,string(sprintf('jdata\n\b\ashall\tprevail')),sprintf('[SU<21>jdata\n\b\ashall\tprevail]'),'debug',1);
        test_jsonlab('string array',@savebj,[string('jdata');string('shall');string('prevail')],'[[SU<5>jdataSU<5>shallSU<7>prevail]]','debug',1);
    end
    test_jsonlab('empty name',@savebj,loadbj(['{U' 0 'U' 2 '}']),'{U<0>U<2>}','debug',1);
    if(exist('containers.Map'))
        test_jsonlab('empty name with map',@savebj,loadbj(['{U' 0 'U' 2 '}'],'usemap',1),'{U<0>U<2>}','debug',1);
    end
    test_jsonlab('row vector',@savebj,[1,2,3],'[$U#U<3><1><2><3>','debug',1);
    test_jsonlab('column vector',@savebj,[1;2;3],'[$U#[$U#U<2><3><1><1><2><3>','debug',1);
    test_jsonlab('mixed array',@savebj,{'a',1,0.9},'[CaU<1>D<0.9>]','debug',1);
    test_jsonlab('char array',@savebj,['AC';'EG'],'[SU<2>ACSU<2>EG]','debug',1);
    test_jsonlab('maps',@savebj,struct('a',1,'b','test'),'{U<1>aU<1>U<1>bSU<4>test}','debug',1);
    test_jsonlab('2d array',@savebj,[1,2,3;4,5,6],'[$U#[$U#U<2><2><3><1><2><3><4><5><6>','debug',1);
    test_jsonlab('3d (row-major) nested array',@savebj,reshape(1:(2*3*2),2,3,2),...
         '[[[U<1>U<7>][U<3>U<9>][U<5>U<11>]][[U<2>U<8>][U<4>U<10>][U<6>U<12>]]]','debug',1,'nestarray',1);
    test_jsonlab('3d (column-major) nested array',@savebj,reshape(1:(2*3*2),2,3,2),...
         '[[[U<1>U<2>][U<3>U<4>][U<5>U<6>]][[U<7>U<8>][U<9>U<10>][U<11>U<12>]]]','debug',1,'nestarray',1,'formatversion',1.9);
    test_jsonlab('3d annotated array',@savebj,reshape(int8(1:(2*3*2)),2,3,2),...
         '{U<11>_ArrayType_SU<4>int8U<11>_ArraySize_[$U#U<3><2><3><2>U<11>_ArrayData_[$U#U<12><1><7><3><9><5><11><2><8><4><10><6><12>}','debug',1);
    test_jsonlab('complex number',@savebj,single(2+4i),...
         '{U<11>_ArrayType_SU<6>singleU<11>_ArraySize_[$U#U<2><1><1>U<16>_ArrayIsComplex_TU<11>_ArrayData_[$U#[$U#U<2><2><1><2><4>}','debug',1);
    test_jsonlab('empty sparse matrix',@savebj,sparse(2,3),...
         '{U<11>_ArrayType_SU<6>doubleU<11>_ArraySize_[$U#U<2><2><3>U<15>_ArrayIsSparse_TU<11>_ArrayData_Z}','debug',1);
    test_jsonlab('real sparse matrix',@savebj,sparse([0,3,0,1,4]'),...
         '{U<11>_ArrayType_SU<6>doubleU<11>_ArraySize_[$U#U<2><5><1>U<15>_ArrayIsSparse_TU<11>_ArrayData_[$U#[$U#U<2><2><3><2><4><5><3><1><4>}','debug',1);
    test_jsonlab('complex sparse matrix',@savebj,sparse([0,3i,0,1,4i].'),...
         '{U<11>_ArrayType_SU<6>doubleU<11>_ArraySize_[$U#U<2><5><1>U<16>_ArrayIsComplex_TU<15>_ArrayIsSparse_TU<11>_ArrayData_[$U#[$U#U<2><3><3><2><4><5><0><1><0><3><0><4>}','debug',1);
    test_jsonlab('heterogeneous cell',@savebj,{{1,{2,3}},{4,5},{6};{7},{8,9},{10}},...
         '[[[U<1>[U<2>U<3>]][U<4>U<5>][U<6>]][[U<7>][U<8>U<9>][U<10>]]]','debug',1);
    test_jsonlab('struct array',@savebj,repmat(struct('i',1.1,'d','str'),[1,2]),...
         '[{U<1>iD<1.1>U<1>dSU<3>str}{U<1>iD<1.1>U<1>dSU<3>str}]','debug',1);
    test_jsonlab('encoded fieldnames',@savebj,struct(encodevarname('_i'),1,encodevarname('i_'),'str'),...
         '{U<2>_iU<1>U<2>i_SU<3>str}','debug',1);
    if(exist('OCTAVE_VERSION','builtin')~=0)
        test_jsonlab('encoded fieldnames without decoding',@savebj,struct(encodevarname('_i'),1,encodevarname('i_'),'str'),...
             '{U<2>_iU<1>U<2>i_SU<3>str}','debug',1,'UnpackHex',0);
    else
        test_jsonlab('encoded fieldnames without decoding',@savebj,struct(encodevarname('_i'),1,encodevarname('i_'),'str'),...
             '{U<7>x0x5F_iU<1>U<2>i_SU<3>str}','debug',1,'UnpackHex',0);
    end       
    if(exist('containers.Map'))
         test_jsonlab('containers.Map',@savebj,containers.Map({'Andy','^_^'},{true,'-_-'}),...
             '{U<4>AndyTU<3>^_^SU<3>-_-}','debug',1,'usemap',1);
    end
    if(exist('istable'))
         test_jsonlab('simple table',@savebj,table({'Andy','^_^'},{true,'-_-'}),...
             '{U<11>_TableCols_[SU<4>Var1SU<4>Var2]U<11>_TableRows_ZU<14>_TableRecords_[[SU<4>AndySU<3>^_^][TSU<3>-_-]]}','debug',1);
    end
    if(exist('bandwidth'))
         lband=2;
         uband=1;
         a=double(full(spdiags(true(4,lband+uband+1),-uband:lband,3,4)));
         a(a~=0)=find(a);

         test_jsonlab('lower band matrix',@savebj,tril(a),...
             '{U<11>_ArrayType_SU<6>doubleU<11>_ArraySize_[$U#U<2><3><4>U<14>_ArrayZipSize_[$U#U<2><2><3>U<12>_ArrayShape_[SU<9>lowerbandU<1>]U<11>_ArrayData_[$U#U<6><1><5><9><0><2><6>}','debug',1,'usearrayshape',1);
         test_jsonlab('upper band matrix',@savebj,triu(a),...
             '{U<11>_ArrayType_SU<6>doubleU<11>_ArraySize_[$U#U<2><3><4>U<14>_ArrayZipSize_[$U#U<2><3><3>U<12>_ArrayShape_[SU<9>upperbandU<2>]U<11>_ArrayData_[$U#U<9><7><11><0><4><8><12><1><5><9>}','debug',1,'usearrayshape',1);
         test_jsonlab('diag matrix',@savebj,tril(triu(a)),...
             '{U<11>_ArrayType_SU<6>doubleU<11>_ArraySize_[$U#U<2><3><4>U<12>_ArrayShape_SU<4>diagU<11>_ArrayData_[$U#U<3><1><5><9>}','debug',1,'usearrayshape',1);
         test_jsonlab('band matrix',@savebj,a,...
             '{U<11>_ArrayType_SU<6>doubleU<11>_ArraySize_[$U#U<2><3><4>U<14>_ArrayZipSize_[$U#U<2><4><3>U<12>_ArrayShape_[SU<4>bandU<2>U<1>]U<11>_ArrayData_[$U#U<12><7><11><0><4><8><12><1><5><9><0><2><6>}','debug',1,'usearrayshape',1);
         a=a(:,1:3);
         a=uint8(tril(a)+tril(a)');
         test_jsonlab('symmetric band matrix',@savebj,a,...
             '{U<11>_ArrayType_SU<5>uint8U<11>_ArraySize_[$U#U<2><3><3>U<14>_ArrayZipSize_[$U#U<2><2><3>U<12>_ArrayShape_[SU<13>lowersymmbandU<1>]U<11>_ArrayData_[$U#U<6><2><10><18><0><2><6>}','debug',1,'usearrayshape',1);
         a(a==0)=1;
         test_jsonlab('lower triangular matrix',@savebj,tril(a),...
             '{U<11>_ArrayType_SU<5>uint8U<11>_ArraySize_[$U#U<2><3><3>U<12>_ArrayShape_SU<5>lowerU<11>_ArrayData_[$U#U<6><2><2><10><1><6><18>}','debug',1,'usearrayshape',1);
         test_jsonlab('upper triangular matrix',@savebj,triu(a),...
             '{U<11>_ArrayType_SU<5>uint8U<11>_ArraySize_[$U#U<2><3><3>U<12>_ArrayShape_SU<5>upperU<11>_ArrayData_[$U#U<6><2><2><1><10><6><18>}','debug',1,'usearrayshape',1);
    end
    try
        val=zlibencode('test');
        a=uint8(eye(5));
        a(20,1)=1;
        test_jsonlab('zlib/zip compression (level 6)',@savebj,a,...
            '{U<11>_ArrayType_SU<5>uint8U<11>_ArraySize_[$U#U<2><20><5>U<14>_ArrayZipSize_[$U#U<2><1><100>U<14>_ArrayZipType_SU<4>zlibU<14>_ArrayZipData_[$U#U<18><120><156><99><100><0><1><70><28><36><197><0><108><12><0><2><33><0><7>}',...
            'debug',1, 'Compression','zlib','CompressArraySize',0)  % nestarray for 4-D or above is not working
        test_jsonlab('gzip compression (level 6)',@savebj,a,...
            '{U<11>_ArrayType_SU<5>uint8U<11>_ArraySize_[$U#U<2><20><5>U<14>_ArrayZipSize_[$U#U<2><1><100>U<14>_ArrayZipType_SU<4>gzipU<14>_ArrayZipData_[$U#U<30><31><139><8><0><0><0><0><0><0><3><99><100><0><1><70><28><36><197><0><108><12><0><95><87><171><165><100><0><0><0>}',...
            'debug',1, 'Compression','gzip','CompressArraySize',0)  % nestarray for 4-D or above is not working
        test_jsonlab('lzma compression (level 5)',@savebj,a,...
            '{U<11>_ArrayType_SU<5>uint8U<11>_ArraySize_[$U#U<2><20><5>U<14>_ArrayZipSize_[$U#U<2><1><100>U<14>_ArrayZipType_SU<4>lzmaU<14>_ArrayZipData_[$U#U<32><93><0><0><16><0><100><0><0><0><0><0><0><0><0><0><128><61><72><138><187><229><101><33><24><236><49><255><255><251><90><160><0>}',...
            'debug',1, 'Compression','lzma','CompressArraySize',0)  % nestarray for 4-D or above is not working
    catch
    end
end

%%
if(ismember('bjo',tests))
    fprintf(sprintf('%s\n',char(ones(1,79)*61)));
    fprintf('Test Binary JSON function options\n');
    fprintf(sprintf('%s\n',char(ones(1,79)*61)));

    test_jsonlab('remove ubjson optimized array header',@savebj,[1,2,3],'[U<1>U<2>U<3>]','debug',1,'nestarray',1);
    test_jsonlab('limit to ubjson signed integer',@savebj,256,'I<256>','debug',1,'ubjson',1);
    test_jsonlab('limit to ubjson integer markers',@savebj,2^32-1,'L<4294967295>','debug',1,'ubjson',1);
    test_jsonlab('H marker for out of bound integer',@savebj,2^64-1,'HU<20>18446744073709551616','debug',1,'ubjson',1);
    test_jsonlab('do not downcast integers to the shortest format',@savebj,int32(5),'l<5>','debug',1,'keeptype',1);
    test_jsonlab('do not downcast integer array to the shortest format',@savebj,int32([5,6]),'[$l#U<2><5><6>','debug',1,'keeptype',1);
    test_jsonlab('test little endian uint32',@savebj,typecast(uint8('abcd'),'uint32'),'mabcd','endian','L');
    test_jsonlab('test big endian uint32',@savebj,typecast(uint8('abcd'),'uint32'),'mdcba','endian','B');
    test_jsonlab('test little endian double',@savebj,typecast(uint8('01234567'),'double'),'D01234567','endian','L');
    test_jsonlab('test big endian double',@savebj,typecast(uint8('01234567'),'double'),'D76543210','endian','B');
    test_jsonlab('test default int endian for savebj',@savebj,typecast(uint8('jd'),'uint16'),'ujd');
    test_jsonlab('test default int endian for saveubjson',@saveubjson,typecast(uint8('jd'),'uint16'),'Idj');
    test_jsonlab('test default float endian for savebj',@savebj,typecast(uint8('1e05'),'single'),'d1e05');
    test_jsonlab('test default float endian for saveubjson',@saveubjson,typecast(uint8('12345678'),'double'),'D87654321');
end

%%
if(ismember('jmap',tests))
    fprintf(sprintf('%s\n',char(ones(1,79)*61)));
    fprintf('Test JSON mmap\n');
    fprintf(sprintf('%s\n',char(ones(1,79)*61)));

    test_jsonlab('mmap of a 1D numerical array',@savejson,loadjson('[1,2,3]','mmaponly',1),'[["$",[1,7]]]','compact',1);
    test_jsonlab('mmap of a 1D mixed array',@savejson,loadjson('[1,"2",3]','mmaponly',1),'[["$",[1,9]]]','compact',1);
    test_jsonlab('mmap of a 2D array',@savejson,loadjson('[[1,2,3],[4,5,6]]','mmaponly',1),'[["$",[1,17]]]','compact',1);
    test_jsonlab('mmap of concatenated json',@savejson,loadjson('[1,2,3][4,5,6]','mmaponly',1),'[["$",[1,7]],["$1",[8,7]]]','compact',1);
    test_jsonlab('mmap of concatenated json objects',@savejson,loadjson('[1,2,3]{"a":[4,5]}','mmaponly',1),'[["$",[1,7]],["$1",[8,11]],["$1.a",[13,5]]]','compact',1);
    test_jsonlab('mmap of an array with an object',@savejson,loadjson('[1,2,{"a":3}]','mmaponly',1),...
        '[["$",[1,13]],["$[0]",[2,1]],["$[1]",[4,1]],["$[2]",[6,7]],["$[2].a",[11,1]]]','compact',1);
    test_jsonlab('mmap of an object',@savejson,loadjson('{"a":1,"b":[2,3]}','mmaponly',1),...
        '[["$",[1,17]],["$.a",[6,1]],["$.b",[12,5]]]','compact',1);
    test_jsonlab('mmap of object with white-space',@savejson,loadjson('{"a":1 , "b"  :  [2,3]}','mmaponly',1),...
        '[["$",[1,23]],["$.a",[6,1]],["$.b",[18,5,2]]]','compact',1);
    test_jsonlab('mmapinclude option',@savejson,loadjson('[[1,2,3],{"a":[4,5]}]','mmaponly',1,'mmapinclude','.a'),...
        '[["$[1].a",[15,5]]]','compact',1);
    test_jsonlab('mmapexclude option',@savejson,loadjson('[[1,2,3],{"a":[4,5]}]','mmaponly',1,'mmapexclude',{'[0]','[1]','[2]'}),...
        '[["$",[1,21]]]','compact',1);
    test_jsonlab('json with indentation',@savejson,loadjson(savejson({[1,2,3],struct('a',[4,5])}),'mmaponly',1,'mmapinclude','.a'),...
        '[["$[1].a",[22,7]]]','compact',1);
end

%%
if(ismember('bmap',tests))
    fprintf(sprintf('%s\n',char(ones(1,79)*61)));
    fprintf('Test Binary JSON mmap\n');
    fprintf(sprintf('%s\n',char(ones(1,79)*61)));

    test_jsonlab('mmap of a 1D numerical array',@savejson,loadbj(savebj([1,2,3]),'mmaponly',1),'[["$",[1,9]]]','compact',1);
    test_jsonlab('mmap of a 1D mixed array',@savejson,loadbj(savebj({1,'2',3}),'mmaponly',1),'[["$",[1,8]],["$[0]",[2,2]],["$[1]",[4,2]],["$[2]",[6,2]]]','compact',1);
    test_jsonlab('mmap of a 2D array',@savejson,loadbj(savebj([[1,2,3],[4,5,6]]),'mmaponly',1),'[["$",[1,12]]]','compact',1);
    test_jsonlab('mmap of an array with an object',@savejson,loadbj(savebj({1,2,struct('a',3)}),'mmaponly',1),...
        '[["$",[1,13]],["$[0]",[2,2]],["$[1]",[4,2]],["$[2]",[6,7]],["$[2].a",[10,2]]]','compact',1);
    test_jsonlab('mmap of an object',@savejson,loadbj(savebj(struct('a',1,'b',[2,3])),'mmaponly',1),...
        '[["$",[1,18]],["$.a",[5,2]],["$.b",[10,8]]]','compact',1);
    test_jsonlab('mmapinclude option',@savejson,loadbj(savebj({[1,2,3],struct('a',[4,5])}),'mmaponly',1,'mmapinclude','.a'),...
        '[["$[1].a",[15,8]]]','compact',1);
    test_jsonlab('mmapexclude option',@savejson,loadbj(savebj({[1,2,3],struct('a',[4,5])}),'mmaponly',1,'mmapexclude',{'[0]','[1]','[2]'}),...
        '[["$",[1,24]]]','compact',1);
    test_jsonlab('test multiple root objects with N padding',@savejson,loadbj([savebj({[1,2,3],struct('a',[4,5])}) 'NNN' savebj(struct('b',[4,5]))],'mmaponly',1,'mmapinclude','.b'),...
        '[["$1.b",[32,8]]]','compact',1);
end

%%
if(ismember('bugs',tests))
    fprintf(sprintf('%s\n',char(ones(1,79)*61)));
    fprintf('Test bug fixes\n');
    fprintf(sprintf('%s\n',char(ones(1,79)*61)));
    test_jsonlab('simplify cell arrays mixing numbers and chars',@savejson,loadjson('[1,0,"-","L",900]'),'[1,0,"-","L",900]','compact',1);
    test_jsonlab('simplify cell arrays with string elements',@savejson,loadjson('["j","s","o","n"]'),'["j","s","o","n"]','compact',1);
end