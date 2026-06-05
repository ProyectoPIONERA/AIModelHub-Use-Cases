#!/bin/bash                                                                                                                                                 
                                                                                                                                                            
                                                                                                                             
# Modelos a entrenar                                                                                                                                        
FLARES_MODELS=(                                                                                                                                                    
    "dccuchile/distilbert-base-spanish-uncased"                                                                                                             
    "dccuchile/albert-base-spanish"                                                                                                                         
    #"dccuchile/bert-base-spanish-wwm-uncased"                                                                                                               
)                                                                                                                                                           
                                                                                                                                                                                                                                                                                                                     
# Ejecutar la función main() para cada modelo                                                                                                               
for MODEL in "${FLARES_MODELS[@]}"; do                                                                                                                             
    echo "Entrenando modelo: $MODEL"                                                                                                                        
                                                                                                                                                                                                                                                                                          
                                                                                                                                                            
    # Ejecutar la función main() con los parámetros adecuados                                                                                               
    python -m src.utils.flares.train_models --model ${MODEL}                                                                                                                            

                                                                                                                                                                                                                                                                                    
done

echo "Entrenando y evaluando modelos de movilidad"

python -m src.utils.mobility.train_models --target actual_travel_time

python -m src.utils.mobility.train_models --target delay

python -m src.utils.mobility.train_models --target previous_delay

echo "Entrenamiento y evaluación completados!"  