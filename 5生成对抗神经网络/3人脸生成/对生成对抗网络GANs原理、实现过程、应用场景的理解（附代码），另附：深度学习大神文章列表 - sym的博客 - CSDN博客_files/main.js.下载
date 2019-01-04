/*
*
*  Push Notifications codelab
*  Copyright 2015 Google Inc. All rights reserved.
*
*  Licensed under the Apache License, Version 2.0 (the "License");
*  you may not use this file except in compliance with the License.
*  You may obtain a copy of the License at
*
*      https://www.apache.org/licenses/LICENSE-2.0
*
*  Unless required by applicable law or agreed to in writing, software
*  distributed under the License is distributed on an "AS IS" BASIS,
*  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
*  See the License for the specific language governing permissions and
*  limitations under the License
*
*/

/* eslint-env browser, es6 */

'use strict';
(function(){
  const applicationServerPublicKey = 'BCYaMwiS92AJlv9Eg2YXSFwuI3ppbydkz31gOI5NS7YtOp05n7qUHEyb_iijzQcjgWqrsGSj2K18F21G9DYL4-U';

 // const pushButton = document.querySelector('.js-push-btn');

  let isSubscribed = false;
  let swRegistration = null;
  function urlB64ToUint8Array(base64String) {
    const padding = '='.repeat((4 - base64String.length % 4) % 4);
    const base64 = (base64String + padding)
      .replace(/\-/g, '+')
      .replace(/_/g, '/');

    const rawData = window.atob(base64);
    const outputArray = new Uint8Array(rawData.length);

    for (let i = 0; i < rawData.length; ++i) {
      outputArray[i] = rawData.charCodeAt(i);
    }
    return outputArray;
  }


  if ('serviceWorker' in navigator && 'PushManager' in window) {
    // 浏览器支持serviceworker并且可以进行推送的前提
    // console.log('Service Worker and Push is supported');
    // console.log('浏览器支持serviceworker推送')
// 支持的话 注册sw服务
    // 默认刷新询问
    var hasSub = getCookie('hasSub')
    // 判断域名的操作 如果域名是blog 和download
    var flagBox1 = window.location.href.indexOf('blog')>-1 ?'true':'false'
    var flagBox2 = window.location.href.indexOf('download')>-1 ?'true':'false'

    if((hasSub!=='true'&&flagBox1 === 'true')||(hasSub!=='true'&&flagBox2 === 'true')){
      window.Notifier.RequestPermission();
      var _hmt = _hmt || [];
      // _hmt.push(['_trackEvent', "授权弹框展示", "授权框"]);
      _hmt.push(['_setUserTag', '5703', '授权弹框展示']);

      navigator.serviceWorker.register('/sw.js')
        .then(function(swReg) {
          // console.log('Service Worker is registered', swReg);
          // console.log('Service Worker注册了', swReg);
          swRegistration = swReg;
          // sw注册后调用检查用户是否订阅通知的函数
          initialiseUI();
        })
        .catch(function(error) {
          // console.error('Service Worker Error', error);
          // console.error('Service Worker 报错了', error);
          // _hmt.push(['_trackEvent', "授权弹框展示但由于浏览器不支持报错", "授权框展示浏览器不支持"]);
          _hmt.push(['_setUserTag', '5703', '授权弹框展示但由于浏览器不支持报错']);
        });
    }
  } else {
    // console.warn('Push messaging is not supported');
    // console.log('浏览器不支持serviceworker推送')
    // 按钮提示 推送不支持
    // pushButton.textContent = 'Push Not Supported';
  }

// 检查用户当前有没有订阅
  function initialiseUI() {
    // 设置按钮监听的作用是防止二次点击
    // pushButton.addEventListener('click', function() {
    //   pushButton.disabled = true;
    //   // 点击后就不能再点
    //   if (isSubscribed) {
    //     // 已经订阅过了 点击按钮就取消订阅了
    //     unsubscribeUser();
    //   } else {
    //     // 进行订阅的操作
    //     console.log('每次刷新都走初始化订阅222')
    //     subscribeUser();
    //   }
    // });

    // 初始化触发是否订阅的操作
    // if (isSubscribed) {
    //   // 已经订阅过了 下次进入就取消订阅吗
    //   console.log('初始化取消订阅')
    //   unsubscribeUser();
    // } else {
    //   // 进行订阅的操作
    //   console.log('每次刷新都走初始化订阅')
    //   subscribeUser();
    // }
    //
    // // Set the initial subscription value
    // 执行完订阅的操作后sw 就知道订阅过 还是没有订阅过 然后更新
    swRegistration.pushManager.getSubscription()
      .then(function(subscription) {
        isSubscribed = !(subscription === null);
        // subscription === null true 没有订阅过
        // 向应用服务端发送请求
        updateSubscriptionOnServer(subscription);
        // hasSub：全栈只要有一个域名订阅了 不管当下域名是否订阅 都不再订阅弹出授权
        // isSubscribed 当前域名订阅过不再订阅
        // if (isSubscribed || hasSub === 'true') {
        // console.log('flaging====',flaging)
        if (isSubscribed) {
          // console.log('User IS subscribed.');
          // console.log('用户订阅过了不会再订阅了 也不会取消订阅');
          // 跳转到其他地址会不会再走else一次 再测试一下 如果还会走 那只能用localstorage判断

        } else {
          // console.log('User is NOT subscribed.');
          // console.log('用户没有订阅过 执行订阅逻辑');
         // _hmt.push(['_trackEvent', "选择授权操作", "用户同意授权"]);
          _hmt.push(['_setUserTag', '5703', '用户选择授权操作']);

          subscribeUser();
        }

        updateBtn();
      });
  }
// cookie
  function setCookie(cname, cvalue, exdays) {
    var d = new Date();
    d.setTime(d.getTime() + (exdays * 24 * 60 * 60 * 1000));
    var expires = "expires=" + d.toUTCString();
    document.cookie = cname + "=" + cvalue + "; " + expires+";domain=csdn.net;path=/"
    // console.log(d)
  }


  //获取cookie
  function getCookie(cname) {
    var name = cname + "=";
    var ca = document.cookie.split(';');
    for(var i = 0; i < ca.length; i++) {
      var c = ca[i];
      while(c.charAt(0) == ' ') c = c.substring(1);
      if(c.indexOf(name) != -1) return c.substring(name.length, c.length);
    }
    return "";
  }
//启用我们的按钮，以及更改用户是否订阅的文本
  function updateBtn() {
    if (Notification.permission === 'denied') {
      // pushButton.textContent = '用户选择不授权 推送消息被阻塞 按钮不可点击';
      // pushButton.disabled = true;
      // console.log('用户授权拒绝了 给服务端发请求')
     // _hmt.push(['_trackEvent', "拒绝授权操作", "用户拒绝授权"]);
      _hmt.push(['_setUserTag', '5703', '拒绝授权操作']);
      updateSubscriptionOnServer(null);
      return;
    }
    if (isSubscribed) {
      // 如果已经订阅过了 那么按钮就不可点击了
     // pushButton.textContent = '点击取消订阅';
     //  console.log('updateBtn已经订阅过')
    } else {
     // pushButton.textContent = '可以点击订阅';
     //  console.log('updateBtn没有订阅过')
    }
    //pushButton.disabled = false;
  }
// 点击了按钮 执行订阅的逻辑
  function subscribeUser() {
    const applicationServerKey = urlB64ToUint8Array(applicationServerPublicKey);
    // 传递公钥给sw服务器
    swRegistration.pushManager.subscribe({
      userVisibleOnly: true,
      // 默认允许订阅后发送通知
      applicationServerKey: applicationServerKey
    })
      .then(function(subscription) {
        //console.log('User is subscribed:', subscription);
        // console.log('用户点击订阅已经向sw服务器发送公钥成功 订阅成功',subscription)
        //创建异步对象
        var xhr;
        if (window.XMLHttpRequest)
        {
          //  IE7+, Firefox, Chrome, Opera, Safari 浏览器执行代码
          xhr = new XMLHttpRequest();
        }
        else
        {
          // IE6, IE5 浏览器执行代码
          xhr = new ActiveXObject("Microsoft.XMLHTTP");
        }
//设置请求的类型及url
        // application/x-www-form-urlencoded
        xhr.open('post', 'https://gw.csdn.net/cui-service/v1/browse_info/save_browse_info' );
//发送请求
        //post请求一定要添加请求头才行不然会报错 open后send前
        xhr.setRequestHeader("Content-type","application/json");
        var jsonData ={'subscription':JSON.stringify(subscription)}
        xhr.withCredentials = true;
        xhr.send(JSON.stringify(jsonData));
        xhr.onreadystatechange = function () {
          // 这步为判断服务器是否正确响应
          if (xhr.readyState == 4 && xhr.status == 200) {
            // console.log(xhr.responseText);
          }
        };
        updateSubscriptionOnServer(subscription);

        isSubscribed = true;
        setCookie('hasSub',true,1)
        updateBtn();
        // 更新按钮状态与文案
      })
      .catch(function(err) {
        // console.log('Failed to subscribe the user: ', err);
        // console.log('用户点击订阅已经向服务器发送公钥失败 订阅失败',err)
        updateBtn();
      });
  }
  function unsubscribeUser() {
    // 取消订阅 先获取当前的订阅消息
    swRegistration.pushManager.getSubscription()
      .then(function(subscription) {
        if (subscription) {
          // 调用取消订阅
          return subscription.unsubscribe();
        }
      })
      .catch(function(error) {
        // console.log('Error unsubscribing', error);
        // console.log('调用取消订阅报错',error)
      })
      .then(function() {
        updateSubscriptionOnServer(null);

        // console.log('User is unsubscribed.');
        // console.log('用户已经取消订阅了 更新按钮状态')
        isSubscribed = false;

        updateBtn();
      });
  }

  function updateSubscriptionOnServer(subscription) {
    // TODO: Send subscription to application server
// 打印出订阅的消息 这部分真实逻辑是向服务端发送请求 告诉服务端相应内容
//     console.log('updateSubscriptionOnServer函数是发送给服务端的相应数据subscription',subscription)
    // 可以通过使用服务工作线程中的推送事件，我们可以使用 DevTools 触发虚假的推送事件，测试收到消息后会发生什么在您的网络应用中，订阅推送消息（确保控制台中有 User IS subscribed），然后转至 DevTools 中的 Application 面板，并在 Service Workers 选项卡下，点击相应服务工作线程下的 Push 链接。
  }


}())
