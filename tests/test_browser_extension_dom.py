from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


def test_content_script_observes_only_visible_for_you_posts():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node is required for the isolated extension DOM fixture")
    script = r"""
    const fs = require('node:fs');
    const vm = require('node:vm');
    const assert = require('node:assert/strict');
    let handler;
    let visibility = 'visible';
    let label = 'Для вас';
    const articles = [article('123456', 100), article('789012', 1200)];
    function article(id, top) {
      return {
        getBoundingClientRect: () => ({top, bottom: top + 100, width: 400}),
        querySelector: (selector) => selector === 'time[datetime]' ? {
          closest: () => ({getAttribute: () => `/karpathy/status/${id}`}),
          getAttribute: () => '2026-09-07T08:00:00Z',
        } : selector === '[data-testid="tweetText"]' ? {textContent: 'New agent release'} : null,
        querySelectorAll: () => [{getAttribute: () => 'https://example.com/paper?id=42&utm_source=x'}],
      };
    }
    const context = {
      location: {origin: 'https://x.com', pathname: '/home'}, window: {innerHeight: 800}, URL,
      document: {
        get visibilityState() { return visibility; },
        querySelector: selector => selector.startsWith('[role="tab"]')
          ? {textContent: label} : null,
        querySelectorAll: () => articles,
      },
      chrome: {runtime: {id: 'ours', onMessage: {addListener: fn => {handler = fn;}}}},
    };
    vm.runInNewContext(fs.readFileSync(process.argv[1], 'utf8'), context);
    function ask(type, sender = 'ours') {
      let result;
      handler({type, maxItems: 80}, {id: sender}, value => {result = value;});
      return result;
    }
    assert.equal(ask('OSVED_PROBE').ok, true);
    const snapshot = ask('OSVED_VISIBLE_SNAPSHOT');
    assert.equal(snapshot.result.posts.length, 1);
    assert.equal(snapshot.result.posts[0].id, '123456');
    assert.equal(snapshot.result.posts[0].links[0], 'https://example.com/paper?id=42');
    assert.equal(ask('OSVED_VISIBLE_SNAPSHOT', 'other'), undefined);
    visibility = 'hidden';
    assert.equal(ask('OSVED_VISIBLE_SNAPSHOT').error, 'no_visible_tab');
    visibility = 'visible'; label = 'Following';
    assert.equal(ask('OSVED_VISIBLE_SNAPSHOT').error, 'wrong_feed');
    label = 'For you'; context.location.pathname = '/messages';
    assert.equal(ask('OSVED_VISIBLE_SNAPSHOT').error, 'wrong_feed');
    """
    path = Path(__file__).parents[1] / "browser-extension/content.js"
    subprocess.run([node, "-e", script, str(path)], check=True, capture_output=True, text=True)
