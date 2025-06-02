import com.squareup.moshi.*

sealed interface BriloopValue

data class BriloopValueInt(
    val value: Int,
) : BriloopValue

data class BriloopValueDouble(
    val value: Double,
) : BriloopValue

data class BriloopValueBoolean(
    val value: Boolean,
) : BriloopValue


class BriloopValueAdapter {

    @FromJson
    fun fromJson(
        reader: JsonReader,
    ) : BriloopValue? {
        val peek = reader.peek()
        return when (peek) {
            JsonReader.Token.BOOLEAN -> BriloopValueBoolean(reader.nextBoolean())
            JsonReader.Token.NUMBER -> {
                try {
                    BriloopValueInt(reader.nextInt())
                } catch (e: JsonDataException) {
                    BriloopValueDouble(reader.nextDouble())
                }
            }
            else -> null
        }
    }

    @ToJson
    fun toJson(
        writer: JsonWriter,
        value: BriloopValue?,
    ) {
        when (value) {
            is BriloopValueInt -> writer.value(value.value)
            is BriloopValueBoolean -> writer.value(value.value)
            is BriloopValueDouble -> writer.value(value.value)
            else -> writer.nullValue()
        }
    }
}

