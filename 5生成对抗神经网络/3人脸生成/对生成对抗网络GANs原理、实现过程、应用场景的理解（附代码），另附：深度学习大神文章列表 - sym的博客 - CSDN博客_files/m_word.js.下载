/*
 * 内文提词
 */


(function(){

    var Word = function(config){

        this.config = this.mix({
            di: '',
            id: '',
            user: 'zhidao',
            logid: "111",
            sid: '',
            isMobile: true,
            title: document.title,
            page_url: location.href,
            classname: 'baidu-highlight',
            ignore: 'baidu-gh-highlight-no', // 需要忽略掉的区域
            frequency: 3,//词频
            maxword: 20//最大接受词语个数
        },config);

        this.initElement(typeof this.config.id === 'string'
            ? this.config.id.split(','): this.config.id);
        // 词条个数
        this.wordnum = 0;
        this.frequencyCache = {};

        this.request();
    };
    Word.prototype = {
        cacheFilter: {},
        highnum: 0,
        highwords: {},
        mix: function(t,s){
            for(var p in s){
                t[p]=s[p];
            }
            return t;
        },
        jsonp: function(url){

            var head = document.getElementsByTagName("head")[0]||document.body,
                script = document.createElement('script'),
                self = this;

            window[self.cid] = function(json){
                self.response(json);

                //删除回调函数
                try {
                    delete window[self.cid]
                }catch(e){}
            };

            script.charset = document.charset || document.characterSet || 'gb2312';
            script.src = url;

            try {
              head.appendChild(script);
            } catch (e) {}
        },
        param: function(){
            var param =[];
            for( var p in this.config ) {
                param.push(p + '=' + encodeURIComponent(this.config[p]));
            }

            return param;
        },
        initElement: function(id){
            var i =0,l=id.length,
                e,self = this,element;

            self.element = [];

            for(;i<l;i++) {
                if ( element = document.getElementById(id[i]) ) {
                    self.element.push(element)
                }
            }
        },
        match: function(){
            var cache = {};

            return function(text,word){

                return cache[word] || (cache[word] = new RegExp('(' + word.join('|') + ')', 'ig'));
            }
        }(),
        safeReg: function(regtext){
            return regtext.replace(/(\+|\\|\/|\[|\]|\{|\}|\*|\?|\$|\^|\||\(|\)|\.)/g,function($1,$2){
                return '\\' + $1;
            })
        },
        hasClassName: function (elem, className) {
            if (elem.className == undefined
                 || (typeof elem.className === 'string'
                 && elem.className.indexOf(className) === -1)) {
                return false;
            }
            return true;
        },
        highlight: function (node, wordlist) {

            if ( node.nodeType == 1) {

                if (node.nodeName !== 'A') {

                    if ( node.hasChildNodes ) {

                        var childs = node.childNodes;
                        var idx = 0;
                        var elem;
                        while (elem = childs[idx++]) {
                            if (!this.hasClassName(elem, this.config.ignore)) {
                                this.highlight(elem, wordlist);
                            }
                        }
                    }
                }

            } else if (node.nodeType === 3 && !this.cacheFilter[node.value]) {

                var nodeValue = this.filter(node.nodeValue);
                var parent    = node.parentNode;
                var self      = this;
                var index  = 0;


                var match = self.match(nodeValue, wordlist);

                if ( nodeValue.match(match) ) {

                    self.cacheFilter[node.value]
                    nodeValue = nodeValue.replace(match, function (m) {
                        var key = m.toLowerCase();

                        if (!self.frequencyCache[key]) {
                            self.frequencyCache[key] = 1;
                            self.wordnum ++;
                        }

                        if (self.wordnum > self.config.maxword) {
                            return m;
                        }

                        if (self.frequencyCache[key]++ <= self.config.frequency) {
                            self.highnum++;
                            if (self.highwords[key]) {
                                self.highwords[key]++
                            } else {
                                self.highwords[key] = 1;
                            }
                            return '<a href="' + self.getHref(m) + '" target="_blank" class="'
                                + self.config.classname + '" >' + m + '</a>';
                        } else {
                            return m;
                        }
                    });
                }


                try {
                    var div = document.createElement('div');
                    var childs = div.childNodes;
                    div.innerHTML = nodeValue;

                    for(var i=0,l=childs.length;i<l;i++){
                        var child = childs[i].cloneNode(true);

                        if ( child.nodeType ==1 ){

                            if (child.className == self.config.classname){

                                child.onclick = function(){

                                    self.log({
                                        fm:"click",
                                        r_type:"text",
                                        r_wd: this.innerHTML
                                    });
                                };
                            }
                        }
                        parent.insertBefore(child,node);
                    }

                    parent.removeChild(node);
                }catch(e){}
            }

        },
        filter: function (text) {

            return text.replace(/</ig, '&lt;')
                .replace(/>/ig, '&gt;');
        },

        getProtocol: function(){
            var userAgent = (navigator&&navigator.userAgent? navigator.userAgent:'');
            if ( userAgent.match(/chrome|firefox|safari|msie 10|rsv:11|msie [89]/i) ) {
                return 'https:'
            }
            return 'http:';
        },
        getHref: function(word){
            var protocol = this.getProtocol(),
                paraminfo = this.paraminfo,
                param = [];

            if (this.config.isMobile && paraminfo.r_from) {
                param.push('from='+paraminfo.r_from);
            }

            if (!this.config.isMobile && paraminfo.r_tn) {
                param.push('tn='+paraminfo.r_tn);
            }
            if (paraminfo.r_tu === 'pu100048' && paraminfo.fenlei) {
                param.push('fenlei=' + encodeURIComponent(paraminfo.fenlei));
            }
            if (this.config.isMobile) {
                if (paraminfo.sa) {
                    param.push('sa=' + paraminfo.sa);
                }
                return protocol + '//m.baidu.com/s?word='+encodeURIComponent(word) +'&'+ param.join('&');
            } else {
                if (paraminfo.rsv_dl) {
                    param.push('rsv_dl=' + paraminfo.rsv_dl);
                }
                return protocol + '//www.baidu.com/s?wd='+encodeURIComponent(word) +'&'+ param.join('&');
            }

        },
        render: function(wordlist){
            var self = this;
            var el = this.element;
            var words = [];
            var ei = el.length;
            var word;

            this.cache = {};
            //渲染数据
            for( var i=0,l=wordlist.length;i<l;i++ ) {
                word = wordlist[i].word;
                this.cache[word] = wordlist[i];
                this.highwords[word] = 0;
                words.push(this.safeReg(word));
            }

            for (var e = 0; e < ei; e++) {

                this.highlight(this.element[e], words);
            }


            var arr = [];
            for (var i in self.highwords) {
                arr.push(i+'='+self.highwords[i]);

            }
            // load日志
            if (self.highnum) {
                self.log({
                    fm: 'load',
                    r_type: 'wordner_load',
                    highnum: self.highnum,
                    highwords: arr.join(' ')
                });
            }
        },
        log: function( param ){

            var config = this.mix({
                pid:201,
                productline:'cun',
                productsub:'cun',
                pj:'ext_hl',
                tab:'query',
                logactionid:"0100100002",
                mod:'cun',
                logtype: 'vgif',
                t: +new Date,
                r_tn: this.paraminfo.r_tn,
                r_tu: this.paraminfo.r_tu
            },this.paraminfo),ps=[],img = new Image;

            if ( param.fm=='click' ) {

                config = this.mix(config,this.cache[param.r_wd]||{});
            }

            config = this.mix(config,param);

            for( var p in config ){
                ps.push(p+'='+encodeURIComponent(config[p]))
            }

            img.src = (window.DSPUIDOMINURI||(location.protocol + '//entry.baidu.com')) +'/rp/v.gif?' + ps.join('&');
        },
        response: function(json){
            try{
                if ( (json.err_no=='0') && json.word_list) {
                    if ( json.word_list.length>0 ){
                        var wordlist = json.word_list;
                         //   wordlist = wordlist.splice(0,parseInt(this.config.maxword||'',10));

                        this.paraminfo = json.param_info;

                        this.render(wordlist.length > this.config.maxlength
                            ? wordlist.splice(0,this.config.maxlength): wordlist);
                    }
                }

            } catch(e){

                window.console && console.log(e);
            }
        },
        request: function(){
            var param = this.param(),
                uid   = +new Date,
                self  = this;

            param.push('jsonp='+(self.cid='baidu_hh_'+uid));
            param.push('r_new=true');

            self.jsonp(((window.tc?window.tc.DSPUIIFRAMEURI:window.DSPUIIFRAMEURI)
                    || (location.protocol + '//entry.baidu.com/rp/wordner'))
                     + '?' + param.join('&') + '&mobile=true');
        }
    }
    if (window.tc && window.tc.baidu_hh_hotword) {
        new Word(window.tc.baidu_hh_hotword);
    }else{
        new Word(window.baidu_hh_hotword ||{});
    }
})();
