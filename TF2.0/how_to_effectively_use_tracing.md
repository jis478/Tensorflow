# re-tracing을 방지하기 위한 @tf.function 사용법 에 대한 정리


#### 본 내용은 Cutmix tensorflow 코드를 작성 중에 @tf.function을 사용했는데 계속 re-tracing이 일어나서, 이에 대한 원인과 해결방안을 정리한 글 임 

#### 참고자료 
1. @tf.function 정의
https://www.tensorflow.org/guide/function

2. tracing
https://books.google.co.kr/books?id=HHetDwAAQBAJ&pg=PA793&lpg=PA793&dq=tracing+tensorflow&source=bl&ots=0KwhZriiPu&sig=ACfU3U3z8-bxD9qp_N_AupnnO6fHIur8QA&hl=en&sa=X&ved=2ahUKEwi2oKCFyZ7qAhUSK6YKHWEZA804ChDoATAFegQICxAB#v=onepage&q=tracing%20tensorflow&f=false

## 1. tracing 이란?
기본적으로 주어진 @tf.function decorator가 둘러싸고 있는 python 함수를 tf 함수로 변경해주는 작업 (즉, Tensorflow 1.x 처럼 static graph로 만들어주는 과정)이다. 
Tensorflow 2.x에서는 eager mode가 기본이기 때문에, 1.x 처럼 static 방식 (define-and-run 방법)이 아닌 dynamic (define-by-run 방법) 으로 그래프가 생성이 된다. 따라서 기존 static 방식이
자랑으로 내세웠던 빠른 속도를 일정부분 포기하는 단점이 있었고, 이를 극복하는 방법으로 @tf.function으로 static의 장점을 유지하는 방법인 것이다 (라고 홈페이지에서 강조한다.)

하지만 홈페이지에서 하나 강조하는 것은 @tf.function을 무작정 사용할 경우 충돌이 일어나거나 비효율적인 tracing이 발생할 수 있기 때문에 속도 증가를 못 보는 경우가 발생할 수 있다. 속도 증가는 커녕 side effects가 발생할 수 있다는 것이다. 사실 이러한 이유로 @tf.function 코드를 custom training에 이해 없이 쓰기가 어렵게 느껴진다.

이번에 Cutmix 코드를 짜면서 custom training에 @tf.function을 써봤는데, 오류를 직접 겪으면서 체득한 것이 있어서 기록을 하고자 한다.

일단,  @tf.function으로 감싼 함수를 호출하면, tracing이 일어나며 tensorflow static graph 형태의 concrete_function이 생성이 된다. 재미있는 것은 다음과 같은 룰인데, 당연히 이러한 기능을 쓰는 이유는 위에서 언급한대로 static -> dynamic 함수로의 변환을 원한 것이지만, 이러한 변환이 함수 호출 마다 일어난다면 매우 곤란하게 된다. 이는 python과 tensorflow의 다른 인자 처리 방법에 기인하는데, python 함수는 함수 인자로 다양한 형태를 받을 수 있지만 tensorflow는 static graph일 경우 정해진 형태의 인자만 받을 수 있기 때문이다 (당연한 얘기인듯?)

다음과 같은 예시가 홈페이지에 있다. 여기서 함수 인자로 들어오는 값의 형태가 계속 바뀌기 때문에 (integer -> float -> string), 함수는 호출 시마다 새로 static 그래프를 생성하는 re-tracing을 일으키게 된다. 

#### re-tracing 발생 예제
```
@tf.function
def double(a):
  print("Tracing with", a)
  return a + a

print(double(tf.constant(1)))
print()
print(double(tf.constant(1.1)))
print()
print(double(tf.constant("a")))
print()
```

#### re-tracing 방지 룰
re-tracing을 방지하기 위해 다음과 같은 룰을 지켜야 한다.

1. 인자가 tensor 라면, 동일한 dtype, shape의 tensor인 경우에는 re-tracing이 일어나지 않는다.
2. 인자가 tensor 라면, 만약 다른 dtype, shape의 tensor이거나, 인자가 tensor가 아닌 python value 인 경우에는 다른 value가 들어올 경우 re-tracing이 일어난다. 


#### Cutmix 코드에 적용 
위의 룰을 조금 고민해보면, @tf.function으로 custom training 함수를 감싸는 경우에는 함수로 전달되는 인자들이 python 형태라면 안된다는 결론에 도달할 수 있다. 내가 짠 코드에서도 계속 re-tracing이 난 이유도, cutmix 에서는 lambda 값을 인자로 함수에 전달해줘야 하는데 lambda가 tensor가 아닌 python float32 형태 였기 때문에 문제가 발생한 것이었다. 따라서 lambda 값을 tf.convert_to_tensor로 tensor로 변환 후 수행하면 문제 없이 돌아가는 것을 볼 수 있다. 추가로, @tf.function을 쓸 경우 CIFAR100 기준으로 Epoch 당 학습 시간이 100초 -> 60초로 40% 감소되는 장점을 확인할 수 있었다.

추가로 정리한 점은, 
- 기본적으로 @tf.function으로 감쌀 경우, static 그래프로 변하기 떄문에 이 안에서 기존에 정의한 tensor (eager tensor 포함) 들은 모두 symbolic tensor (numpy 값이 없음)로 변하게 된다.
- 만약 print를 그래프 상에 넣어서 call마다 발생하고 싶으면, tf.print로 수행해야 한다.

```
@tf.function
def train_cutmix_image(image_cutmix, target_a, target_b, lam):
  with tf.GradientTape() as tape:
    output = model(image_cutmix, training=True) 
    loss = criterion(target_a, output) * lam + criterion(target_b, output) * (1. - lam)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables)) 
  return loss, output
  
lam = tf.convert_to_tensor(1 - ((bbx2 - bbx1) * (bby2 - bby1) / (image_cutmix.shape[1] * image_cutmix.shape[2])), dtype=tf.float32)
loss, output = train_cutmix_image(image_cutmix, target_a, target_b, lam)       
```        

## 2. 결론
- 기본적으로 성능 향상을 위해 @tf.function을 쓰는 경우 (Cutmix 예제 참조), 감싸는 함수에 들어가는 인지가 어떤 형태인지 주의를 해야한다.
- custom training 구조를 어떻게 해야만 성능을 극대화 할 수 있을지 더 공부해보자.


