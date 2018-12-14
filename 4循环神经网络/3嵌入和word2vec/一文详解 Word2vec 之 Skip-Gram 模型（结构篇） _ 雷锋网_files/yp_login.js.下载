(function(w, d) {

	var YP_LOGIN_WINDOW = {};
	var G = {};
	var closeButton = null;

	G.hasClass = function(o, c) {
		var regexp = new RegExp("(^|\\s+)" + c + '(\\s+|$)');
		return regexp.test(o.className);
	}
	G.addClass = function(o, c) {
		!G.hasClass(o, c) && (o.className += ' ' + c);
	}
	G.removeClass = function(o, c) {
		o.className = o.className.replace(new RegExp("(^|\\s+)" + c + '(\\s+|$)'), ' ');
	}

	function scriptOnload(node, callback) {
		if ("onload" in node) {
			node.onload = onload;
			node.onerror = function() {
				throw new Error('加载失败');
			}
		} else {
			node.onreadystatechange = function() {
				if (/loaded|complete/.test(node.readyState)) {
					onload();
				}
			}
		}

		function onload() {
			node.onload = node.onerror = node.onreadystatechange = null;
			node = null;
			callback();
		}
	}

	//异步加载js和css文件
	function require(url, callback, css) {
			var head = d.head || d.getElementsByTagName("head")[0] || d.documentElement;
			var baseElement = head.getElementsByTagName("base")[0];
			var currentlyAddingScript;
			if (css) {
				var node = document.createElement('link');
			} else {
				var node = d.createElement("script");
			}
			scriptOnload(node, callback);
			currentlyAddingScript = node;
			baseElement ? head.insertBefore(node, baseElement) : head.appendChild(node);
			if (css) {
				node.rel = "stylesheet"
				node.href = url;
			} else {
				node.async = true;
				node.src = url;
			}
			currentlyAddingScript = null;
		}
		//创建iframe
	function createIframe() {
		var frm = document.createElement('iframe');
		//var ref = window.location.protocol + '//' +window.location.host;
		var ref = HOME_URL;
		frm.id = 'YP_LOGIN_WINDOW';
		frm.className = 'YP-LOGIN';
		frm.frameborder = 0;
		frm.scrolling = "no";
		d.body.appendChild(frm);
		return frm;
	}

	// 创建遮罩
	function createOverlay() {
		var ov = document.createElement('div');
		closeButton = document.createElement('div');
		ov.id = 'YP_LOGIN_WINDOW_OVERLAY';
		ov.className = 'YP-OVERLAY';
		closeButton.className = 'close';
		closeButton.title = '关闭';
		closeButton.style.backgroundImage = 'url(' + USER_CENTER + 'resWeb/home/images/login/close_03.png)';
		ov.appendChild(closeButton);
		d.body.appendChild(ov);
		return ov;
	}

	// 获取当前站点的域名
	function getSiteName (){
		var host = window.location.host;
		var sites = ['wankr100','leiphone','knewbi','igao7'];
		for(var i = 0; i < sites.length; i++){
			if(host.indexOf(sites[i]) != -1){
				return sites[i];
			}
		}
		return sites[0]; // 默认玩客
	}

	// 入口
	function main() {

		require(USER_CENTER + '/resWeb/home/css/login/yp_login.css', function() {}, true);
		var frm = createIframe();
		var overlay = createOverlay();

		function t(){
			var wh = window.innerHeight;
			var sh = frm.offsetHeight || 478;

			return wh - sh <  0 ? 0 : (wh - sh) / 3;
		}

		var t1;

		YP_LOGIN_WINDOW.show = function() {
            var ref = HOME_URL;
            frm.src = USER_CENTER +'main/remoteLogin?url=' + encodeURIComponent(ref)+'&returnUrl='+encodeURIComponent(location.href) + '&site=' + getSiteName();
			$(overlay).fadeIn(350);
			t1 = t();
			$(frm).css({top : t1 - 50 , display : 'block' , opacity : 0}).animate({
				top : t1,
				opacity : 1
			});

			//frm.style.top = t() + 'px';
		}
		YP_LOGIN_WINDOW.hide = function() {
    
            $(overlay).fadeOut(350);
			$(frm).animate({
				top : t1 - 50,
				opacity : 0
			},function(){
				$(this).css('display','none');
			});
		}

		//微信首次登录
		YP_LOGIN_WINDOW.weixin_first_login_complement = function(){
			if(/#wechat_first_login/.test(window.location.href) && ($('#is_login_tag_status').val() < 1 || $('#is_login_tag_status').length < 1)){
				var ref = HOME_URL;
            	frm.src = USER_CENTER +'main/wechatOauthLogin?url=' + encodeURIComponent(ref)+'&returnUrl='+encodeURIComponent(location.href) + '&site=' + getSiteName();
				$(overlay).css('display','block');
				$(frm).css({top : t() , display : 'block'});
                window.location.href = window.location.href.replace(/#wechat_first_login/,'#');
			}
		}
		//注册框
		YP_LOGIN_WINDOW.register_show = function(){
			var ref = HOME_URL;
            //frm.src = USER_CENTER +'main/quickRegister?url=' + encodeURIComponent(ref)+'&returnUrl='+encodeURIComponent(location.href) + '&site=' + getSiteName();
            frm.src = USER_CENTER +'login/wechatLogin?url=' + encodeURIComponent(ref)+'&returnUrl='+encodeURIComponent(location.href) + '&site=' + getSiteName();
			$(overlay).fadeIn(350);
			t1 = t();
			$(frm).css({top : t1 - 50 , display : 'block' , opacity : 0}).animate({
				top : t1,
				opacity : 1
			});
		}

		overlay.onclick = function(){
			YP_LOGIN_WINDOW.hide();
		}


		
		//微信登录成功后
		YP_LOGIN_WINDOW.callback_show = function(){
			
			var ref = HOME_URL;
            frm.src = USER_CENTER +'login/UserCenterCallback?url=' + encodeURIComponent(ref)+'&returnUrl='+encodeURIComponent(location.href) + '&site=' + getSiteName();
			$(overlay).fadeIn(350);
			t1 = t();
			$(frm).css({top : t1 - 50 , display : 'block' , opacity : 0}).animate({
				top : t1,
				opacity : 1
			});
		
		}
		

	}
	/*&
	if(typeof jQuery == 'undefined'){
		main();
	}else{
		require(USER_CENTER + 'js/jquery.js',function(){
			main();
		});
	}
	*/
	main();

	//判断地址栏参数
	var hrefStr = window.location.href;
	if(hrefStr.indexOf("userCenterCallback")>-1){
		YP_LOGIN_WINDOW.callback_show();
	}

	YP_LOGIN_WINDOW.weixin_first_login_complement();

	w['YP_LOGIN_WINDOW'] = YP_LOGIN_WINDOW;


	


})(window, document);
