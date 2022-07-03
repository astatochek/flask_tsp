# Flask TSP Solver
## Описание
Реализация на основе `Flask` нескольких алгоритмов на графах с визуализацией.

Для того, чтобы работать с браузерным приложением, требуется установить фреймворк `flask` и библиотеки `matplotlib`,  `numpy` и некоторые другие, запустить файл `app.py` и пройти по ссылке на сайт, которая появится в консоли.

## Пример работы
| Plane Graph      |      3D Graph     |   Graph on a Sphere   |
| :---:        |    :----:   |          :---: |
|<img src="https://i.imgur.com/oE3WlkY.png">|<img src="https://imgur.com/wmtmzBs.gif">|<img src="https://imgur.com/IACQ4P7.gif">|
|<img src="https://imgur.com/D32rS4f.png" width=300>|<img src="https://imgur.com/goLefqc.gif" width=300> | <img src="https://imgur.com/rkBpdKu.gif" width=300> |
|<img src="https://imgur.com/hwv5lNZ.png" width=300>|<img src="https://imgur.com/GRP56WU.gif" width=300> | <img src="https://imgur.com/QRqif8z.gif" width=300> |
|<img src="https://imgur.com/WzT1JUY.png" width=300>|<img src="https://imgur.com/2YCS91M.gif" width=300> | <img src="https://imgur.com/qiYttOY.gif" width=300> |

## Функционал
Реалищовано 2 алгоритма для решения задачи Коммивояжера: жадный и 2-приближенный. Жадный работает за $O(n^3)$: для каждой вершины он запускает обычный жадный поиск перестановки, начинающейся в данной вершине, а из полученных $n$ выбирает лучшую. Приближенный алгоритм находит минимальное остовное дерево, удваивает ребра и строит Эйлеров обход полученного графа. Также реализовано построение минимального остовного дерева.

## Визуализация

Для 2-мерного случая берутся либо случайно расположенные в квадрате $[0, 100]\times[0, 100]$ точки, либо расположенные на окружности радиуса 100. Тогда выводятся `.png` изображения, построенные посредством `matplotlib`. В 3-мерном случае реализована анимация (`.gif` из 80-и кадров), полученная последовательным построением графов с поворотом вокруг оси Z. Это сильно сказывается на верремени работы: независимо от количества узлов ждать приходится от 30 до 40 минут для вывода всех четырех изображений. Случайные точки берутся из параллелепипеда $[-100, 100] \times [-100, 100]\times [0, 100] $. Граф на сфере получается на основе выбора двух случайных углов $\varphi , \psi \in [0, 2\pi)$ и полученных на основе них координат, за расстояние берется кратчайшее расстояние на между точками на повержности сферы радиуса 100.