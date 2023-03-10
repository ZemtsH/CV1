# CV1
<h1 align="center">Медианный фильтр</h1>
<h2>Алгоритм</h2>
<p>Чтобы выполнить процесс фильтрации изображений, нужен фильтр , также называемый маской . 
Этот фильтр обычно представляет собой двумерное квадратное окно, 
то есть окно с равными размерами (шириной и высотой).</p>
<img src = "https://docs.gimp.org/2.10/ru/images/filters/examples/blur/median-calcul.png">
<p>Фильтр скользит по массиву сигнала, и возвращает на каждом шаге один из элементов, попавших в окно (апертуру) фильтра. Выходной сигнал yk скользящего медианного фильтра шириной n для текущего отсчета k формируется из входного временного ряда …, xk-1, xk, xk+1,… в соответствии с формулой:</p>
<p>yk = Me(xk-(n-1)/2,…, xk,…,xk+(n-1)/2),</p>
<p>где Me(x1,…,xn) = x((n+1)/2) – элементы вариационного ряда, т.е. ранжированные в порядке возрастания значений x1 = min(x1,…, xn) ≤ x(2) ≤ x(3) ≤ … ≤ xn = max(x1,…, xn). Ширина медианного фильтра выбирается с учетом того, что он способен подавить импульс шириной (n-1)/2 отсчетов, при условии, что n – нечетное число.</p>
<p>Таким образом, медианная фильтрация реализуется в виде процедуры локальной обработки отсчетов в скользящем окне, которое включает определенное число отсчетов сигнала. Для каждого положения окна выделенные в нем отсчеты ранжируются по возрастанию или убыванию значений. Средний по своему положению отсчет в ранжированном списке называется медианой рассматриваемой группы отсчетов, если число отсчетов нечетно. Этим отсчетом заменяется центральный отсчет в окне для обрабатываемого сигнала. При четном количестве отсчетов медиана устанавливается, как среднее арифметическое двух средних отсчетов. В качестве начальных и конечных условий фильтрации обычно принимается текущее значение сигнала, либо медиана находится только для тех точек, которые вписываются в пределы апертуры.</p>
<p>Благодаря свои характеристикам, медианные фильтры при оптимально выбранной апертуре могут сохранять без искажений резкие границы объектов, подавляя некоррелированные и слабо коррелированные помехи и малоразмерные детали. В аналогичных условиях алгоритмы линейной фильтрации неизбежно «смазывают» резкие границы и контуры объектов.</p>

<h2>Разработка</h2>
<p>Для воспроизведения видео из файла используем функцию cv.VideoCapture</p>
<p>Для вывода изображения в окно - cv2.imshow()</p>
<h3>Реализация с использованием встроенных функций библиотеки OpenCV</h3>
<p>Для медианной фильтрации просто вызываем функцию cv2.medianBlur()</p>
<p>Время обработки кадра составляет ~ 0.079 секунды.</p>
<p>Время обработки всего видео ~ 3.884 секунд.</p>
<h3>Нативно</h3>
<p>Для начала определяем крайнее значение, ширину и высоту. Создаем пустой массив. Создаем окно фильтра,соответствующее заданному размеру. С помощью циклов проходим по всем пикселям применяя медианный фильтр.</p>
<p>Время обработки кадра составляет ~ 52.395 секунд.</p>
<p>Время обработки всего видео ~ 2567.389 секунд.</p>
<h3>Нативно с использованием Numba</h3>
<p>Что бы ускорить функцию используем библиотке numba. Перед определением функции вписываем декоратор njit</p>
<p>Время обработки кадра составляет ~ 2.417 секунд.</p>
<p>Время обработки всего видео ~ 118.442 секунд.</p>
<h2>Тестирование</h2>
<p>Исходное изображение</p>
<img src = "https://github.com/ZemtsH/CV1/blob/main/img/1.jpg?raw=true">
<p>Изображение после обработки</p>
<img src = "https://github.com/ZemtsH/CV1/blob/main/img/2.jpg?raw=true">
<h2>Выводы по работе</h2>
<p>При использовании библиотеки OpenCV для Python обработка изображения проходит гораздо быстрее. Даже при использовании библиотек для ускорения функций, обработка видео требует много ресурс.</p>
