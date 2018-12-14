(function(window,document/*,undefined*/){
  //2013.1.24 by 司徒正美
  function contains(parentEl, el, container) {
      // 第一个节点是否包含第二个节点
    //contains 方法支持情况：chrome+ firefox9+ ie5+, opera9.64+(估计从9.0+),safari5.1.7+
    if (parentEl == el) {
      return true;
    }
    if (!el || !el.nodeType || el.nodeType != 1) {
      return false;
    }
    if (parentEl.contains ) {
      return parentEl.contains(el);
    }
    if ( parentEl.compareDocumentPosition ) {
      return !!(parentEl.compareDocumentPosition(el) & 16);
    }
    var prEl = el.parentNode;
    while(prEl && prEl != container) {
      if (prEl == parentEl)
          return true;
      prEl = prEl.parentNode;
    }
    return false;
  }

  /**
   * 转换obj为字符串param
   */
  function parseParam(params) {
    if(!params) return '';
    var sb = "", val;
    for(var key in params) {
      val = params[key];
      sb += ("&" + key + "=" + encodeURIComponent(val));
    }
    return sb;
  }

  /**
   * 获取特殊类型的物料
   */
  function getCSDNAdsItem() {
    var adsItem = {
      item_id: "user_define",
      special_type: 'csdn_net_alliance_ads',
      ads: 1
    };
    return adsItem;
  }

  function randomStr(len){
    var sb="";var dict="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefghijklmnopqrstuvwxyz";
    for(var i=0;i<len;++i)sb+=dict.charAt(Math.random()*dict.length|0);return sb;
  }

  function hrTime(x){
    var date=new Date(x*1000),it;
    var MM=(it=date.getMonth()+1)<10?'0'+it:it;
    var dd=(it=date.getDay())<10?'0'+it:it;
    var HH=(it=date.getHours())<10?'0'+it:it;
    var mm=(it=date.getMinutes())<10?'0'+it:it;
    var ss=(it=date.getSeconds())<10?'0'+it:it;
    return date.getFullYear()+'-'+MM+'-'+dd+' '+HH+':'+mm+':'+ss;
  }

  function newXHR(stack){
    var xhr=new XMLHttpRequest();if(!window.TINGYUN||!TINGYUN.createEvent)return xhr;
    var event=TINGYUN.createEvent({name:stack.join('_'),key:"b3d532c8-f6e2-4978-8b7f-31e7255c46e9"});
    xhr.addEventListener("error",function(){event.fail();});
    xhr.addEventListener("load",function(){event.end();});
    return xhr;
  }

  /**
   * action @param {string} 动作类型
   * sceneID @param {string} 场景ID
   * itemPfx @param {string} 场景前缀
   * itemID @param {string} 物料Id
   * context @param {string} 物料的上下文
   * isNetAllianceAds @param {boolean} 如果是网盟广告类型要添加额外字段
   */
  function action(action,sceneID,itemPfx,itemID,context) {
    var url=host+"/action/api/log?requestID="+requestID+"&clientToken="+clientToken,
      ref= {
        "requestID": requestID, "actionTime": Date.now(), "action": action,
        "sceneId": sceneID, "userId": userID, "itemId": itemID,
        "context": context,"itemSetId": "" + itemPfx,"uuid_tt_dd":uuid_tt_dd,
      };
    var xhr=newXHR(["p4sdk","log",sceneID,itemPfx]);

    // 如果是网盟信息的话添加额外字段到ref当中作为标记
    if(itemID === "user_define") {
      ref.specialType = 'csdn_net_alliance_ads',
      ref.ads = 1
    }

    xhr.open("POST",url);
    xhr.setRequestHeader("Content-Type", "text/plain");
    xhr.send(JSON.stringify({
      "date": hrTime(ref.actionTime/1000),
      "actions": [ref]
    }));
  }

  if(window["p4sdk_singleton_main"]) return;
  window["p4sdk_singleton_main"] = true;

  var host="https://nbrecsys.4paradigm.com";
  var uuid_tt_dd = (document.cookie.match(/\buuid_tt_dd=([^;]+)/)||[])[1];
  var clientToken="1f9d3d10b0ab404e86c2e61a935d3888";
  var k="paradigmLocalStorageUserIdKey";var userID=localStorage[k]||(localStorage[k]=randomStr(10));
  var requestID=randomStr(8);var seedItemID=(location.href.match(/\/article\/details\/(\d+)/)||[])[1];
  var req={itemID:seedItemID,uuid_tt_dd:uuid_tt_dd};
  var url=host+"/api/v0/recom/recall?requestID="+requestID+"&userID="+userID+"&sceneID=";

  action("detailPageShow",34,42,seedItemID);var dedup={};

  // T3位置的渲染
  function p4CSDNT3Bootstrap(csdnRender, div) {
    var xhr=newXHR(["p4sdk","recall",34]);
    xhr.open("POST",url+34);
    xhr.addEventListener("load",function(){
      var raw=xhr.responseText;var json=JSON.parse(raw);var item=json[0];
      if(item) {
        if(dedup[item["item_id"]]) item=json[1];
        if(item) dedup[item["item_id"]]=1;
      }

      // 这里我们需要创造一个特殊的物料，用来标明物料类型为：csdn的网盟广告
      if(!item) item = getCSDNAdsItem();
      csdnRender(item, div);

      // 只有对用户可见的时候才上报
      setTimeout(scroll);
      window.addEventListener("scroll", scroll);
      function scroll() {
        if(!contains(document, div)) return;
        var rect = div.getBoundingClientRect();
        var x=(rect.left+rect.right)/2, y=(rect.top+rect.bottom)/2;
        if(x>=0&&x<=document.documentElement.clientWidth&&y>=0&&y<=document.documentElement.clientHeight){
          action("show",34,39,item["item_id"],item["context"]);
          window.removeEventListener("scroll",scroll);
        }
      }

      // 点击上报
      div.addEventListener("click",function() {
        action("detailPageShow",34,39,item["item_id"],item["context"]);
        // var sep=item.url.indexOf('?')<0?'?':'&'; // TODO add before hash(#)
        // window.open(item.url+sep+"utm_source=blogre1" + paramsStr,"_blank");
      });
    });
    xhr.send(JSON.stringify(req));
  }
  window.p4CSDNT3Bootstrap = p4CSDNT3Bootstrap;

  function sampleBuildUrl(stage,itemID,url){
    var sb=host+"/api/v0/csdn/sample-t0?stage="+stage+"&requestID="+requestID+
      "&userID="+userID+"&itemID="+itemID;if(url)sb+="&url="+encodeURIComponent(url);return sb;
  }

  // T0位推荐的渲染 window["p4sdk_enable_courseBox"]
  function p4CSDNT0Bootstrap(csdnRender, div) {
    var xhr = newXHR(["p4sdk","recall",420]);
    xhr.open("POST", url + 420);
    xhr.addEventListener("load",function(){
      var raw=xhr.responseText;var json=JSON.parse(raw);var item=json[0];
      if(item) {
        if(dedup[item["item_id"]]) item=json[1];
        if(item) dedup[item["item_id"]]=1;
      }

      // 这里我们需要创造一个特殊的物料，用来标明物料类型为：csdn的网盟广告
      if(!item) item = getCSDNAdsItem();
      switch(item["item_id"]|0){
        case 143:case 142:case 141:case 140:case 187:
          item["_url"]=item["url"];var a=item["item_id"];
          var b=sampleBuildUrl(2,a);b=b.substring(b.indexOf('?')+1);
          item["url"]=sampleBuildUrl(1,a,item["_url"])//+"#"+b;
      }
      csdnRender(item, div);

      // 只有推荐位对用户可见时才对用户课件
      setTimeout(scroll);
      window.addEventListener("scroll",scroll);
      function scroll() {
        if(!contains(document, div)) return;
        var rect=div.getBoundingClientRect();var x=(rect.left+rect.right)/2,y=(rect.top+rect.bottom)/2;
        if(x>=0&&x<=document.documentElement.clientWidth&&y>=0&&y<=document.documentElement.clientHeight){
          action("show",420,39,item["item_id"],item["context"]);
          window.removeEventListener("scroll",scroll);
        }
      }

      (div.querySelector("a")||div).addEventListener("click",function() {
        action("detailPageShow",420,39,item["item_id"],item["context"]);
        // var sep=item.url.indexOf('?')<0?'?':'&'; // TODO add before hash(#)
        // var paramsStr = parseParam(params);
        // window.open(item.url + sep + "utm_source=blogt0" + paramsStr, "_blank");
        var b=item["_url"];if(!b)return;var xhr=new XMLHttpRequest();
        xhr.open("GET",sampleBuildUrl(0,item["item_id"],b));xhr.send();
      });
    });
    xhr.send(JSON.stringify(req));
  };
  window.p4CSDNT0Bootstrap = p4CSDNT0Bootstrap;
})(window,document);
