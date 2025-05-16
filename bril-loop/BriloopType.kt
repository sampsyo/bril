import com.squareup.moshi.*

sealed interface BriloopType

data class BriloopPrimitiveType(
    val name: String,
) : BriloopType

data class BriloopParameterizedType(
    val map: Map<String, String>?,
) : BriloopType

class BriloopTypeAdapter {

    @FromJson
    fun fromJson(
        reader: JsonReader,
        mapDelegate: JsonAdapter<Map<String, String>>,
    ): BriloopType? {
        val nextToken = reader.peek()
        return if (nextToken == JsonReader.Token.BEGIN_OBJECT) {
            BriloopParameterizedType(
                map = mapDelegate.fromJson(reader),
            )
        } else if (nextToken == JsonReader.Token.STRING) {
            BriloopPrimitiveType(reader.nextString())
        } else {
             null
        }
    }

    @ToJson
    fun toJson(
        writer: JsonWriter,
        briloopType: BriloopType?,
        mapDelegate: JsonAdapter<Map<String, String>>,
    ) {
        when (briloopType) {
            is BriloopPrimitiveType -> writer.value(briloopType.name)
            is BriloopParameterizedType -> mapDelegate.toJson(writer, briloopType.map)
            else -> writer.nullValue()
        }
    }
}

