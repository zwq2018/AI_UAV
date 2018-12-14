/**
 * @Created by nanda221 on 15-5-13.
 * @use 测试页面的预埋点代码
 */

/**
 * 私有方法集合
 */

window.goldeneye_deploy_domain_server = window.location.href.indexOf('local') !== -1 || window.location.href.indexOf('127.0.0.1') !== -1 ?
    '//127.0.0.1:6001' : (window.location.href.indexOf('daily') !== -1 ?
    '//goldeneye.aliyun.com' :
    '//goldeneye.aliyun.com');

window.goldeneye_deploy_domain_static = window.location.href.indexOf('local') !== -1 || window.location.href.indexOf('127.0.0.1') !== -1 ?
    '//127.0.0.1:8000' : (window.location.href.indexOf('daily') !== -1 ?
    '//g.alicdn.com' :
    '//g.alicdn.com');

window.goldeneye_deploy_namespace = {
    urlConfig: {
        check_distribution: window.goldeneye_deploy_domain_server + '/action/deploy/checkDistribution.json',
        distribution_js: window.location.href.indexOf('local') !== -1 ?
            window.goldeneye_deploy_domain_static + '/aliyun/goldeneye-deploy/0.0.1/static/distribution.js?t=' + new Date().getTime() :
            window.goldeneye_deploy_domain_static + '/aliyun/goldeneye-deploy/0.0.1/static/distribution.js?t=' + new Date().getTime()
    },
    jsonp: function(url, success, callbackName){
        var isIE = /*@cc_on!@*/!1,
            script = document.createElement('script'),
            callbackName = callbackName || 'goldeneye_jsonp_callback';

        function gc(){
            // Handle memory leak in IE
            script.onload = script.onreadystatechange = null;
            if( document.head && script.parentNode ){
                document.head.removeChild(script);
            }
        }
        if(isIE){
            script.onreadystatechange = function(){
                var readyState = this.readyState;
                if(readyState == 'loaded' || readyState == 'complete'){
                    gc();
                }
            }
        }else{
            script.onload = function(){
                gc();
            }
        }

        if(success) {
            window[callbackName] = function(data){success.call(null, data)} || function(){};
            script.src = url + ((url.indexOf('?') != -1) ? '&' : '?') + 'callback=' + callbackName + '&ts=' + (new Date).getTime();
        }
        else{
            //说明是单纯的js加载请求
            script.src = url;
        }
        document.head.appendChild(script);
    },
    deleteParamByKey: function(url, key){
        var arr = key instanceof Array ? key : [key],
            result = url;

        for(var i = 0; i < arr.length; i++){
            var word = arr[i];
            result = result.replace(new RegExp(word + '=[^&]*&', 'ig'), '');
            result = result.replace(new RegExp('&' + word + '=[^&]*', 'ig'), '');
            result = result.replace(new RegExp('\\?' + word + '=[^&]*', 'ig'), '');
        }

        if(result.indexOf('?') == result.length - 1){
            result = result.slice(0, result.length - 1);
        }
        //去尾斜杠
        if(result[result.length - 1] == '/'){
            result = result.slice(0, result.length - 1);
        }
        return result;
    },
    getUrlQueryString: function(name, search){
        var reg = new RegExp("(^|&)"+ name +"=([^&]*)(&|$)");
        var r = (search || window.location.search.substr(1)).match(reg);
        if(r!=null){
            return  r[2];
        }else{
            return null;
        }
    },
    log: function(info){
        window.console && window.console.log && console.log('goldeneye deploy: ' + info);
    },
    clearLS: function(){
        localStorage.removeItem('goldeneye_testId_disStrategy');
        localStorage.removeItem('goldeneye_vid');
    }
};



/**
 * 主逻辑区开始
 */

try {
    //不重复执行预埋脚本
    if(!window.goldeneye_hasDeploy){
        window.goldeneye_hasDeploy = true;

        var gn = window.goldeneye_deploy_namespace;

        if(!window.localStorage){
            gn.log('test abort, localstorage not support');
        }
        else if(window.location.href.indexOf('goldeneye_edit') !== -1){
            gn.log('test abort, test is in edit mode');
        }
        else if(window.location.href.indexOf('goldeneye_heatmap=true') !== -1){
            gn.log('test abort, test is in heatmap mode');
        }
        else{
            window.goldeneye_deploy_testurl = gn.deleteParamByKey(window.location.href, ['spm', 'ge_ut', 'ge_ver', 'tracelog']);
            window.goldeneye_deploy_testid = null;

            //localstorage获取上次的测试id和测试命中信息
            var reqUrl = gn.urlConfig.check_distribution + '?testUrl=' + encodeURIComponent(goldeneye_deploy_testurl),
                testId_disStrategy = localStorage.getItem('goldeneye_testId_disStrategy') || '',
                oldTestId = testId_disStrategy ? testId_disStrategy.split('_')[0] : '',
                olddisStrategy = testId_disStrategy ? testId_disStrategy.split('_')[1] : '',
                urlShunt = '';

            //如果是url测试跳过来的，直接带过来测试id，并标记为url分流类型
            if (window.location.href.indexOf('ge_ut') != -1) {
                oldTestId = gn.getUrlQueryString('ge_ut');
                olddisStrategy = '';
                urlShunt = true;
            }

            reqUrl += '&oldTestId=' + oldTestId + '&olddisStrategy=' + olddisStrategy + '&shunt=' + urlShunt;

            gn.jsonp(reqUrl, function (res) {
                if(res.code === 200) {
                    if (res.data && res.data.testId) {

                        if(res.data.shuntVersion) {
                            window.goldeneye_deploy_urlshunt = res.data.shuntVersion;
                        }
                        else {
                            //如果有测试信息，但测试信息和当前运行测试不一致，则清空测试信息
                            if(res.data.testId != oldTestId){
                                gn.clearLS();
                            }
                            //写测试id和命中信息
                            localStorage.setItem('goldeneye_testId_disStrategy', res.data.testId + '_' + res.data.disStrategy);
                        }

                        //如果测试命中
                        if (res.data.disStrategy == 'HIT') {
                            window.goldeneye_deploy_testid = res.data.testId;
                            //加载测试脚本
                            gn.jsonp(gn.urlConfig.distribution_js);
                        }
                        else{
                            gn.log('not hit');
                        }
                    }
                }
                else{
                    res.message && gn.log(res.message);
                    gn.clearLS();
                }
            }, 'jsonCallback');
        }
    }
} catch (e){
    window.goldeneye_deploy_namespace.log('goldeneye has syntax error in goldeneye.js.');
}
