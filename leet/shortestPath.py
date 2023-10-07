import collections
import heapq
from typing import *


def networkDelayTime(times: List[List[int]], N: int, K: int) -> int:
    graph = collections.defaultdict(list)
    # 그래프 인접 리스트 구성

    for u, v, w in times:
        graph[u].append((v, w))

    # 큐 변수: [(소요시간, 정점)]
    Q = [(0, K)]
    dist = collections.defaultdict(int)

    # 우선순위 큐 최솟값 기준으로 정점까지 최단 경로 삽입
    while Q:
        time, node = heapq.heappop(Q)
        if node not in dist:
            dist[node] = time
            for v, w in graph[node]:
                alt = time + w
                heapq.heappush(Q, (alt, v))

    # 모든 노드의 최단 경로 존재 여부 판별
    if len(dist) == N:
        return max(dist.values())

    return -1


def myNetworkDelayTime(times: List[List[int]], N: int, K: int) -> int:
    graph = collections.defaultdict(list)
    for u, v, w in times:
        graph[u].append((v, w))

    Q = [(0, K)]  # weight, node
    dist = collections.defaultdict(int)  # distance map

    while Q:
        w, v = heapq.heappop(Q)
        if v not in dist:
            dist[v] = w
            for n, w2 in graph[v]:
                heapq.heappush(Q, (w + w2, n))

    print(dist)

    return max(dist.values()) if len(dist) == N else -1


def myFindCheapestPrice(n: int, flights: List[List[int]], src: int, dst: int, k: int) -> int:
    graph = collections.defaultdict(list)
    for s, e, w in flights:
        graph[s].append((e, w))

    visited = set()
    Q = [(0, src, 0)]

    while Q:
        w1, curr, stopover = heapq.heappop(Q)
        if curr == dst:
            return w1
        if stopover <= k and (curr, stopover) not in visited:
            visited.add((curr, stopover))
            for ngbhr, w2 in graph[curr]:
                heapq.heappush(Q, (w1 + w2, ngbhr, stopover + 1))

    return -1


def findCheapestPrice(n: int, flights: List[List[int]], src: int, dst: int, k: int) -> int:
    graph = collections.defaultdict(list)
    for u, v, w in flights:
        graph[u].append((v, w))

    Q = [(0, src, k)]

    while Q:
        price, node, stopover = heapq.heappop(Q)
        if node == dst:
            return price
        if stopover >= 0:
            for v, w in graph[node]:
                alt = price + w
                heapq.heappush(Q, (alt, v, stopover - 1))
    return -1


def myNetworkDelayTime2(times: List[List[int]], n: int, k: int) -> int:
    graph = collections.defaultdict(list)
    for s, e, d in times:
        graph[s].append((e, d))

    Q = [(0, k)]
    dist = {}

    while Q:
        w1, prev_node = heapq.heappop(Q)
        if prev_node not in dist:
            dist[prev_node] = w1

            for n, w2 in graph[prev_node]:
                heapq.heappush(Q, (w1 + w2, n))

    if len(dist) < n:
        return -1

    return max(dist.values())


if __name__ == "__main__":
    # n = 4
    # flights = [[0, 1, 100], [1, 2, 100], [2, 0, 100], [1, 3, 600], [2, 3, 200]]
    # src = 0
    # dst = 3
    # k = 1

    # n = 4
    # flights = [[0, 1, 1], [0, 2, 5], [1, 2, 1], [2, 3, 1]]
    # src = 0
    # dst = 3
    # k = 1

    # n = 5
    # flights = [[0, 1, 5], [1, 2, 5], [0, 3, 2], [3, 1, 2], [1, 4, 1], [4, 2, 1]]
    # src = 0
    # dst = 2
    # k = 2
    #
    # print(findCheapestPrice(n, flights, src, dst, k))
    # print(myFindCheapestPrice(n, flights, src, dst, k))

    # times = [[3, 1, 5], [3, 2, 2], [2, 1, 2], [3, 4, 1], [4, 5, 2], [5, 6, 1], [6, 7, 1], [7, 8, 1], [8, 1, 1]]
    times = [[2, 1, 1], [2, 3, 1], [3, 4, 1]]
    N = 4
    K = 2
    print(myNetworkDelayTime2(times, N, K))
