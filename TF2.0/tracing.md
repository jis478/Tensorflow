#올바른 @tf.function에 대한 정리 (작성 중)

1. @tf.function 정의
https://www.tensorflow.org/guide/function

2. tracing
https://books.google.co.kr/books?id=HHetDwAAQBAJ&pg=PA793&lpg=PA793&dq=tracing+tensorflow&source=bl&ots=0KwhZriiPu&sig=ACfU3U3z8-bxD9qp_N_AupnnO6fHIur8QA&hl=en&sa=X&ved=2ahUKEwi2oKCFyZ7qAhUSK6YKHWEZA804ChDoATAFegQICxAB#v=onepage&q=tracing%20tensorflow&f=false

tracing이란, 기본적으로 주어진 @tf.function decorator가 둘러싸고 있는 python 함수를 tf 함수로 변경해주는 작업 (즉, Tensorflow 1.x 처럼 static graph로 만들어주는 과정)이다. 
Tensorflow 2.x에서는 eager mode가 기본이기 때문에, 1.x 처럼 static 방식 (define-and-run 방법)이 아닌 dynamic (define-by-run 방법) 으로 그래프가 생성이 된다. 따라서 기존 static 방식이
자랑으로 내세웠던 빠른 속도를 일정부분 포기하는 단점이 있었고, 이를 극복하는 방법으로 @tf.function으로 static의 장점을 유지하는 방법인 것이다. 하지만 @tf.function을 무작정 사용할 경우 충돌이 일어나거나 비효율적인 tracing이 발생할 수 있기 때문에 속도 증가를 못 보는 경우가 발생할 수 있다.

@tf.function으로 감싼 함수를 호출하면, tracing이 일어나며 concrete_function이 생성이 된다. 여기서 concrete_function    

@tf.function을 사용할 경우 발생하는 tracing은 다음과 같다.

tracing은 함수를 
tracing은 함수의 인자가 동일한 dtype, shape의 tensor인 경우에는 일어나지 않는다.
하지만, 만약 다른 dtype, shape의 tensor이거나, 인자가 tensor가 아닌 python value 인 경우에는 다른 value가 들어올 경우 tracing이 일어난다. 
따라서, 만약 그래프 생성에 영향을 안미치는 인자를 받은 경우에는 re-tracing이 안 일어나야만 효율적이다. 만약 tracing이 계속 일어난다는 것은 static graph를 계속 다시 생성하는 것이기 때문에 매우 비효율적이기 때문에 문제가 있는 것임.
만약 그래프 생성에 영향을 미치는 인자인경우에는 당연히 tracing이 다시 일어나야 한다. 인자에 따라 그래프가 달라져야 하기 때문.




2. 나의 의견
기본적으로 성능 향상을 위해 @tf.function을 쓰는 경우 (Cutmix 예제 참조), 단순히 decorator를 붙이는게 아니라 @tf.function으로 감싸는 함수에 어떤 파이썬 연산이 있는지를 사전에 고려해야 한다. 
예를 들어, 
